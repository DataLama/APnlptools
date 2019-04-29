# =============================================================================
#  Version: 0.1 (April 29, 2019)
#  Author: 김동욱 (datalama@amorepacific.com), 디지털R&D팀
#
#  Contributors:
#
# =============================================================================
#  This is the Function of AP NLP preprocessing.
# =============================================================================

import sys
import json
import boto3
import subprocess





def list2json(batch, size=52428800): 
    """
    Split huge dataset and generate small json files
    
    * batch : list of dictionary that can be converted into json.
    * size : size of json pieces.(defualt = 50MB)
    """
    
    size_checker = 0 
    fn_num = 0
    piece_list = []
    
    for dic in batch:
        
        if size_checker > size: # 기준 size보다 string이 커지면 저장.
            file_name = f"namu_{fn_num:03d}" 
            with open(f'./namu/{file_name}', 'w') as f:
                f.write('\n'.join(piece_list))
                
            size_checker = 0
            fn_num +=1
            piece_list = []
                
        else: 
            piece = json.dumps(dic)
            size_checker += sys.getsizeof(piece) # history of pieces
            piece_list.append(piece)

    # 마지막 나머지 저장!
    with open(f'./namu/namu_{fn_num:03d}', 'w') as f:
        f.write('\n'.join(piece_list))
        
    print('끝!!!')
    return




def uploads_to_s3(bucket, data_key, directory):
    """
    Upload dir and files to s3
    
    * bucket : s3 bucket
    * datakey : s3 directory where you want to save
    * directory : ec2 local's directory(file들이 저장된 directory... s3에 dir과 file들을 함께 업로드함.)
    """
    
    ## file_names
    fns = subprocess.check_output(['ls', '-l', f'{directory}'],stderr=subprocess.PIPE, universal_newlines=True)
    fns = [x.split()[-1] for x in fns.split('\n')[1:-1]] # file_name_lists
    
    ## s3
    s3 = boto3.resource('s3')
    
    ## uploads
    for file in fns:
        file_name = directory + '/' + file
        with open(file_name, 'rb') as f:
            data = f.read()
            key = data_key + '/' + file_name
            s3.Bucket(bucket).put_object(Key = key, Body = data)

    print('FINISH!!!')
    return

class DatasetFromS3(object):
    """
    S3에서 데이터 끌어와서 SageMaker 또는 Local에 저장하기.
    
    USAGE
    * self.executre(loca_dir = [your_directory])
    """
    def __init__(self, bucket = 'ap.nlp', directory = 'dataset/general_language'):
        self.bucket =  bucket
        self.directory = directory
        self.s3r = boto3.resource('s3')
        self.s3c = boto3.client('s3')
        self.channel_list = [obj.key for obj in self.s3r.Bucket(bucket).objects.all() if (directory in obj.key) & (re.search('(\d)$', obj.key) != None)] # 해당 dir에서 끝 자리가 수로 끝나는 데이터들 추출.
        
    def __repr__(self):
        return f"DatasetFromS3('{self.bucket}', '{self.directory}')"
    
    def __str__(self):
        return f"Get Dataset In {self.bucket} Where {self.directory})"
    
    def _data_loader(self):
        """
        S3에서 데이터를 generator를 활용해 끌어온다.
        * output - list of dictionary
        """
        for channel in self.channel_list:
            obj = self.s3c.get_object(Bucket = self.bucket, Key = channel)
            yield obj['Body'].read().decode('utf-8')
            
    def execute(self, local_dir = 'dataset_raw'):
        ## loader_gen
        self.loader_gen = self._data_loader()
        
        ## safe mkdir
        try:
            os.mkdir(local_dir)
        except OSError:
            pass
        
        ## save iterably
        for i, doc_str in enumerate(self.loader_gen):
            file_name = f"./{local_dir}/data{i:03d}"
            with open(file_name, 'w') as f:
                f.write(doc_str)
                
class TextPreprocessor(object):
    """
    argparse로 scripting할 수 있도록 만들기.
    multiprocessing 가능하도록 만들기.
    """
    def __init__(self, local_dir, beauty = False, sub_pos = True):
        self.local_dir = local_dir
        self.Meaningful_Pos_set = set(['NNG', 'NNP', 'VV', 'VA', 'IC', 'ETN', 'ETM', 'XPN', 'XSN', 'XR'])
        self.beauty = beauty # beauty preprocessing
        self.sub_pos = sub_pos
        self.syn_dict = syn_dict
        
        ## file_name_list
        fns = subprocess.check_output(['ls', '-l', f'{local_dir}'],stderr=subprocess.PIPE, universal_newlines=True)
        self.fns = [x.split()[-1] for x in fns.split('\n')[1:-1]]
        
        
        
    def __repr__(self):
        pass
    
    def __str__(self):
        pass


            
    def _base_preprocessing(self, str_list): 
        str_list = [re.sub(r'http\S+', r'<URL>', text) for text in str_list]
        str_list = [re.sub('[~!@#$%^&*><)(-+_=/|\:;}{].', ' ', text) for text in str_list]
        str_list = [re.sub('[^0-9a-zA-Z가-힣]', ' ', text) for text in str_list]
        str_list = [text.upper() for text in str_list]
        return str_list    
    
    def _tokenizer(self, str_list):
        """
        mecab 사용 ## wrapping된애는 init하지마.. multiprocessing안됨.
        output - list of token list
        """
        mecab = Mecab()
        
        if self.sub_pos and self.beauty: # sub pos
            tokened_str_list  = [mecab.pos(text) for text in str_list]
            return [[token if self.syn_dict.get(token) == None else self.syn_dict.get(token) for token, pos in token_set if pos in self.Meaningful_Pos_set] for token_set in tokened_str_list if token_set != []]
        
        elif self.sub_pos and not self.beauty:
            tokened_str_list  = [mecab.pos(text) for text in str_list]
            return [[token for token, pos in token_set if pos in self.Meaningful_Pos_set] for token_set in tokened_str_list if token_set != []]

        else: # all pos
            return [mecab.morphs(text) for text in str_list]
        
            
    def _work(self, doc):
        """
        1. text에 대하여 split('\n') 적용.(정확한 문장 단위는 아니지만 어느 정도 근접함.)
        2. base_preprocessing
        3. tokenizing
        4. synonym 처리.
        """        
        
        if self.beauty:
            return
        else:
            return self._tokenizer(self._base_preprocessing(doc['content'].split('\n')))
    

    def w2v_token_gen(self, workers = 1):
        """
        multiprocessing
        _data_loader => _base_preprocessing => _tokenizer => _synonym_replace
        """

        for file in self.fns:
            file_name = self.local_dir + '/' + file
            with open(file_name,'r') as f:
                doc_list = [json.loads(dic) for dic in f.readlines()]
            
            with Pool(processes = workers) as pool:
                yield itertools.chain(*pool.map(self._work, doc_list)) # 3중 list => 2중 list


            
            
        