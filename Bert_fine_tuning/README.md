# 버트 미세조정 - 슬롯 태깅  
  
1. pretrained BERT 모델을 모듈로 export  
    - ETRI에서 사전훈련한 BERT의 체크포인트를 가지고 BERT 모듈을 만드는 과정.  
    - `python export_korbert/bert_to_module.py -i {체크포인트 디렉토리} -o {output 디렉토리}`   
    - 예시: `python export_korbert/bert_to_module.py -i "/content/drive/MyDrive/004_bert_eojeol_tensorflow" -o "/content/drive/MyDrive/bert-module"`  
  
2. 데이터 준비  
    - 모델을 훈련하기 위해 필요한 seq.in, seq.out이라는 2가지 파일을 만드는 과정.  
    - `python prepare_data.py -i {input파일} -o {output 디렉토리} -vp {(프리트레인 모듈경로)/assets/vocab.korean.rawtext.list}`   
    - 예시: `python prepare_data.py -i "/content/drive/MyDrive/data/sample_data.txt" -o "/content/drive/MyDrive/data/sample/" -vp "/content/drive/MyDrive/bert/assets/vocab.korean.rawtext.list"`  
  
3. Fine-tuning 훈련  
    - 프리트레인된 BERT 모듈에 Fine-tuning 레이어를 추가하여 슬롯 인식이 되도록 하는 과정
    - `python train_bert_finetuning.py -bp {Pre-trained BERT module 경로} -t {훈련 데이터 경로} -v {검증 데이터 경로} -s {Fine-tuned BERT model 저장 경로} -e {epochs} -bs {batch size} -tp bert`
    - 예시: `python train_bert_finetuning.py -bp "/content/drive/MyDrive/bert-module" -t "./sample/" -v "./sample/validation/" -s "./Fine_tuned" -e 3 -bs 256 -tp bert`
  
4. 모델 평가  
    - 따로 준비된 테스트용 데이터를 이용하여 모델 성능을 평가하는 과정
    - `python eval_bert_finetuned.py -bp {Pre-trained BERT module 경로} -m {Fine-tuned BERT model 경로} -d {테스트 데이터 경로} -tp bert`
    - 예시: `python eval_bert_finetuned.py -bp "./Bert_pretrained/" -m "./Fine_tuned/" -d "./sample/test/" -tp bert `
    - 테스트의 결과는 --model에 넣어준 모델 경로 아래의 `test_results`에 저장된다.  
  
3. Inference (임의의 문장을 모델에 넣어보기)  
    - 훈련 완료돤 모델에 임의의 문장을 넣어서 어떻게 추론하는지 살펴보기
    - 모델 자체가 용량이 커서 불러오는 데까지 시간이 걸림  
    - "Enter your sentence:"라는 문구가 나오면 모델에 넣어보고 싶은 문장을 넣어 주면 됨  
    - quit라는 입력을 넣어 주면 종료  
    - `python inference.py -bp {Pre-trained BERT module 경로} -m {Fine-tuned BERT model 경로} -tp bert`
    - 예시: `python inference.py -bp "./Bert_pretrained/" -m "./Fine_tuned" -tp bert`