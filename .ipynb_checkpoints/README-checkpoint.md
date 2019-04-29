# CONTENTS
- APnlp에 사용되는 다양한 tools.


## Custom.
- list2json : GB단위의 json 데이터 셋(list of dictionary)을 기준 크기에 맞게 쪼개는 함수. 
- uploads_to_s3 : 특정 디렉토리에 있는 파일들을 디렉토리째로 s3로 업로드하는 함수.
- DatasetFromS3 : s3에서 학습에 사용할 데이터셋을 다운로드하는 클래스.
- TextPreprocessor : 모델링에 적합한 데이터셋을 만드는 클래스.


## Wiki data extractor
- 위키피디아.
    - [wikiextractor.py](https://github.com/attardi/wikiextractor)
- 나무위키.
    - [namu-wiki-extractor](https://github.com/jonghwanhyeon/namu-wiki-extractor)