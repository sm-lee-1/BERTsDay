# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import re
import sys
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "Bert_fine_tuning"))
#sys.path.append('web_demo/app/')
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
from sklearn import metrics
import traceback
import requests
import json
from .sms import auth
from .sms import config as smsconfig

load_folder_path = os.path.join(os.getcwd(), "Fine_tuned") # 파인튜닝 경로
bert_model_hub_path = os.path.join(os.getcwd(), "Bert_pretrained") #프리트레인 경로
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
is_bert = True

# 슬롯태깅 모델과 벡터라이저 불러오기
print("===============초기화 중===============")
global graph
graph = tf.get_default_graph()
# this line is to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)
bert_to_array = BERTToArray(is_bert, vocab_file)
with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)
model = BertSlotModel.load(load_folder_path, sess)
tokenizer = FullTokenizer(vocab_file=vocab_file)
print("===============초기화 완료===============")

#로그 쓰기
def writeLog(content: str, level=0):
    logLevel=["INFO", "WARNING", "ERROR", "TRACE"]
    now = datetime.now()
    columns = "#level, time, content\n"
    try:
        #로그 파일의 첫번째 줄 읽기 (제대로 된 로그파일인지 검사하기 위함)
        with open(f"log {now.year}-{now.month}-{now.day}.log", "r+") as f: # 예) log 2021-04-26.log
            first = f.readline()
        #파일은 있는데 첫 줄이 본래 방식과 다르면
        if first != columns:
            with open(f"log {now.year}-{now.month}-{now.day}.log", "w") as f:
                f.write(columns) #처음부터 다시 쓰기
                f.flush()
        #로그 쓰기
        with open(f"log {now.year}-{now.month}-{now.day}.log", "a") as f:
            f.write(f"\"{logLevel[level]}\", \"{now.year}-{now.month}-{now.day} {now.hour}:{now.minute}:{now.second}.{now.microsecond}\", \"{content}\"\n")
            f.flush()
    except FileNotFoundError as e:
        #파일이 없으면 파일 새로 만들기
        f = open(f"log {now.year}-{now.month}-{now.day}.log", "a")
    except Exception as e:
        print("로그 파일 작성 중 오류 발생:", e)

# 플라스크 앱 초기화
app = Flask("BERTsDay Chatbot")
app.static_folder = 'web_demo/app/static'
app.template_folder = "web_demo/app/templates"
#run_with_ngrok(app)
writeLog("서버 시작", 0)



@app.route("/")
def home():
# 슬롯 사전 만들기
    app.slot_dict = {'start': '', 'end': '', 'date': '', 'person': '', 'name': '', 'phone': ''}
    app.question = "all"
    app.score_limit = 0.7

    return render_template("index.html")

greeting_arr = ['안녕하세요', '안녕하세요.', '안녕하세요~', '안녕', '안녕~', '안녕!', '안뇽']

answer_first_arr = ['안녕하세요! 스터디룸 예약 챗봇입니다. 이렇게 방문해 주셔서 감사합니다 이제 예약을 도와드릴게요',
	                '안녕하세요. 현재 저희 스터디룸은 코로나19의 확산에 따른 잠재적 전염 위험에 대비하여 정부의 방역 지침에 따라 운영 중에 있습니다. 보다 자세한 사항에 대해서는 시설 내 비치되어 있는 책자에 구체적으로 나와있으니 이에 따라주시면 대단히 감사하겠습니다.',
	                '안녕하세요. 오늘도 저희 스터디룸을 찾아주셔서 감사드립니다.']

answer_last_arr = ['저희 K-107 스터디룸을 이용해주셔서 감사합니다! 예약 당일 본 건물 2층으로 올라오신 후 정면의 문으로 들어오시면 바로 우측에 직원이 있으니 그쪽에서 먼저 결제한 후에 각 방으로 안내를 받으시면 됩니다.	또한 앞서 예약하신 사항에 대해서는 입력하신 번호로 문자를 발송해드리고 있으니 잠시 후 확인 부탁드립니다. 저희 스터디룸을 예약해주셔서 다시 한번 감사합니다∼']

answer_confirm_arr = ['총 몇 명이신가요?',
                     '/이름;/날짜;/번호;/시작시간;/부터/종료시간;/까지로 맞으신가요? 맞다면 이대로 예약 진행하도록 하겠습니다.']

answer_name_arr = ['성함이 어떻게 되시나요?',
                   '이름을 말해주세요.',
                   '예약하시는 분 성함이 어떻게 되시나요?',
                   '예약자님 성함도 한번만 알려주시겠어요?']
answer_phone_arr = ['연락 가능한 번호를 써주세요.(예시 : 010-1234-1234)',
                    '전화번호를 알려주세요.(예시 : 010-1234-1234)',
                    '예약자 분의 번호를 입력해주세요.(예시 : 010-1234-1234)',
                    '연락 가능한 번호 하나만 남겨주시겠어요? (000-0000-0000)',
                    '예약하시는 분께 연락할 수 있는 번호를 남겨주시겠어요? 추후 예약 진행상황을 문자로 알려드릴 예정이니 정확히 기입 부탁드립니다.']
answer_date_arr = ['몇 월 며칠에 예약하고 싶으신가요?',
                   '예약하고 싶은 월일을 입력해주세요. (예시: 1월 3일)',
                   '예약하시려는 날짜를 알려주세요.',
                   '방문하시려는 날짜는 언제인가요?']
answer_start_arr = ['몇 시로 예약하실 건가요?',
                    '몇 시부터 사용하실 건가요?',
                    '사용 시작 시간을 알려주세요.',
                    '도착시간은 언제이신가요? (본 스터디룸은 정각을 기준으로 예약을 받고 있습니다.)',
                    '사용 시작시간은 어떻게 되나요?']
answer_end_arr = ['몇 시까지 이용하실 건가요?',
                  '언제까지 사용하실 건가요?',
                  '종료 시간을 알려주세요.',
                  '사용은 몇시까지 하실 예정이신가요?',
                  '몇시까지 이용할 계획이신가요?']
answer_start_end_arr = ['몇시부터 몇시까지 이용하실 예정이신가요?',
                        '스터디룸 이용은 몇시부터 몇시까지 하실 계획이신가요?']
answer_person_arr = ['총 몇 명이신가요?',
                     '몇 명이서 쓰실 건가요?',
                     '이용 인원을 말씀해주세요?']


date_dict = {'오늘': 0, '금일': 0, '내일': 1, '낼': 1, '모레': 2}
person_dict = {'한명': 1, '혼자': 1, '둘': 2, '두명': 2, '둘이': 2, '셋': 3, '세명': 3, '셋이': 3, '넷': 4, '네명': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8}

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장

    # 날짜에 관련된 문구가 있을때 아래 값으로 대체함
    today = datetime.now()
    for key, value in date_dict.items():
        if key in userText:
            writeLog(f"시간을 표현한 문구가 있어서 값이 대체됨, raw_input: {userText}")
            date_val = today + timedelta(days=value)
            userText = userText.replace(key, str(date_val.month) + "월" + str(date_val.day) + "일")
            

    # 인원에 관련된 문구가 있을때 아래 값으로 대체함
    for key, value in person_dict.items():
        if key in userText:
            writeLog(f"인원에 관련된 문구가 있어서 값이 대체됨, raw_input: {userText}")
            userText = userText.replace(key, str(value) + '명')
            

    #벡터화
    input_text = ' '.join(tokenizer.tokenize(userText))
    token_list = input_text.split()
    data_text_arr = [input_text]
    data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)

    input_date = ''
    input_start = ''
    input_end = ''
    input_person = ''
    input_name = ''
    input_phone = ''
    
    #모델 불러오고 슬롯태깅
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)

    try:
        # 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
        for i in range(0,len(inferred_tags[0])):
            if slots_score[0][i] >= app.score_limit:
                if inferred_tags[0][i]=='날짜':
                    input_date += token_list[i]
                    app.slot_dict['date'] = re.sub('_','',input_date)

                elif inferred_tags[0][i]=='시작시간':
                    if app.question != "end":
                        input_start += token_list[i]
                        temp = input_start.replace("_", "")
                        app.slot_dict['start'] = re.sub('[^0-9]','',temp)
                    else:
                        input_end += token_list[i]
                        temp = input_end.replace("_", "")
                        app.slot_dict['end'] = re.sub('[^0-9]','',temp)

                elif inferred_tags[0][i]=='종료시간':
                    input_end += token_list[i]
                    temp = input_end.replace("_", "")
                    app.slot_dict['end'] = re.sub('[^0-9]','',temp)

                elif inferred_tags[0][i]=='인원':
                    input_person += token_list[i]
                    app.slot_dict['person'] = re.sub('_','',input_person)

                elif inferred_tags[0][i]=='이름':
                    #첫 질문 또는 이름을 물어봤을 때만 채워넣기 (이름 오인식 방지를 위함)
                    if app.question == "name" or app.question == "all": 
                        input_name += token_list[i]
                        app.slot_dict['name'] = re.sub('_','',input_name)

                elif inferred_tags[0][i]=='번호':
                    input_phone += token_list[i]
                    app.slot_dict['phone'] = re.sub('_','',input_phone)

        else: #결과를 다 뽑아냈다면
            # 시작, 종료시간의 끝에 시를 붙임
            if app.slot_dict['start'] != "" and not app.slot_dict['start'].endswith("시"):
                app.slot_dict['start'] += "시"
            if app.slot_dict['end'] != "" and not app.slot_dict['end'].endswith("시"):
                app.slot_dict['end'] += "시"
            # 날짜의 끝에 일을 붙임
            if app.slot_dict['date'] != "" and not app.slot_dict['date'].endswith("일"):
                app.slot_dict['date'] += "일"
        
        # 디버깅용 상태 표시 문장
        if app.debug:
            response = f"<br><br>slot_dict: {app.slot_dict}<br>input_text: {token_list}<br>inferred_tags: {inferred_tags} <br>slots_score: {slots_score}"
        else:
            response = ""

	    # 시작시간 형식 체크
        if app.slot_dict['start'] != '':
            if re.compile(r'^1?[0-9]{1,2}시$').search(app.slot_dict['start']) == None:
                app.slot_dict['start'] = ''
                writeLog(f"잘못된 시간 형식, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return '시작시간 형식이 잘못되었습니다.' + response

        # 종료시간 형식 체크
        if app.slot_dict['end'] != '':
            if re.compile(r'^1?[0-9]{1,2}시$').search(app.slot_dict['end']) == None:
                app.slot_dict['end'] = ''
                writeLog(f"잘못된 시간 형식, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return '종료시간 형식이 잘못되었습니다.' + response

        # 시작시간과 종료시간 체크
        if (app.slot_dict['start'] != '') and (app.slot_dict['end'] != ''):
            start_time = int(re.sub('시', '', app.slot_dict['start']))
            end_time = int(re.sub('시', '', app.slot_dict['end']))

            if start_time < 10:
                start_time += 12
                app.slot_dict['start'] = str(start_time) + '시'

            if end_time < 11:
                end_time += 12
                app.slot_dict['end'] = str(end_time) + '시'

            if start_time > end_time:
                app.slot_dict['end'] = ''
                writeLog(f"잘못된 시작시간-종료시간 범위, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return '종료시간 입력이 잘못되었습니다. 다시 입력해주세요.' + response

        # 날짜 체크 : "x월x일"만 허용
        if app.slot_dict["date"] != "":
            if re.compile(r"[0-9]{1,2}월[0-9]{1,2}일").search(app.slot_dict["date"]) == None:
                app.slot_dict["date"] = ""
                writeLog(f"잘못된 날짜 형식, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return "날짜는 00월 00일 형식으로 입력해주세요. <br>(오늘, 내일도 가능합니다)" +response

        # 인원 체크 : 1~8명까지만 허용
        if app.slot_dict['person'] != '':
            if re.compile(r'^[1-8]명$').search(app.slot_dict['person']) == None:
                app.slot_dict['person'] = ''
                writeLog(f"잘못된 인원(1~8명만 가능) 범위, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return '인원은 1명에서 8명까지 이용 가능합니다.' + response
		
        # 전화번호 형식 체크
        if app.slot_dict['phone'] != '':
            if re.compile(r'^010[0-9]{8}$').search(app.slot_dict['phone']):
                app.slot_dict['phone'] = str(app.slot_dict['phone'][0:3]) + '-' + str(app.slot_dict['phone'][3:7]) + '-' + str(app.slot_dict['phone'][7:11])

            if re.compile(r'^010-[0-9]{4}-[0-9]{4}$').search(app.slot_dict['phone']) == None:
                app.slot_dict['phone'] = ''
                writeLog(f"잘못된 연락처 형식, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}")
                return '연락처 형식이 잘못되었습니다.' + response

        # 이름 형식 체크: 이름이 2~3글자의 한글로만
        if app.slot_dict["name"] != "":
            app.slot_dict["name"] = app.slot_dict["name"].replace("_", "") # 띄어쓰기 공백으로
            searchResult = re.compile(r"[가-힣]{2,3}").search(app.slot_dict["name"]) # 이름으로 추출된 글자들 중에서도 최대 3글자만 추출
            if searchResult == None:# or not len(app.slot_dict["name"]) in (2,3):
                app.slot_dict["name"] = ""
                writeLog(f"잘못된 이름 형식: {userText}")
                app.question = "name"
                return "이름은 2~3글자의 한글로만 작성해주세요" + response
            else:
                app.slot_dict["name"] = searchResult.group()

        # 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
        if ((app.slot_dict['start'] != "") and (app.slot_dict['end'] != "") and (app.slot_dict['person'] != "")and (app.slot_dict['date'] != "") and (app.slot_dict['name'] != "") and (app.slot_dict['phone'] != "")):
            writeLog(f"예약 완료, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}", 0)

            #문자 보내기
            data = {
                'message': {
                    'to': app.slot_dict['phone'].replace("-",""),
                    'from': '01099850384',
                    'text': "[K-107 스터디룸]" +app.slot_dict['name']+'님 ' + app.slot_dict['date'] + ' ' + app.slot_dict['start'] + '부터 ' + app.slot_dict['end'] + '까지 ' + app.slot_dict['person'] + ' 예약되었습니다. ' # 한글 45자, 영어 90자 이상이면 LMS로 자동 발송
                }
            }
            #smsResult = requests.post(smsconfig.getUrl('/messages/v4/send'), headers=auth.get_headers(smsconfig.apiKey, smsconfig.apiSecret), json=data)
            #writeLog("문자 보낸 결과: "+json.dumps(json.loads(smsResult.text), indent=2, ensure_ascii=False))
            #print("문자 보낸 결과: "+json.dumps(json.loads(smsResult.text), indent=2, ensure_ascii=False))
            return app.slot_dict['name']+'님 ' + app.slot_dict['date'] + ' ' + app.slot_dict['start'] + '부터 ' + app.slot_dict['end'] + '까지 ' + app.slot_dict['person'] + ' 예약되었습니다. ' + app.slot_dict['phone'] + '으로 문자 보내드리겠습니다. 감사합니다.' + response
    
        elif ((app.slot_dict['start'] == "") and (app.slot_dict['end'] == "") and (app.slot_dict['person'] == "") and (app.slot_dict['date'] == "") and (app.slot_dict['name'] == "") and (app.slot_dict['phone'] == "")):
            for txt in greeting_arr:
                if txt in userText:
                    writeLog(f"인사말 리턴, raw_input: {userText}", 0)
                    return str(np.random.choice(answer_first_arr, 1)[0])
            if "예약" in userText:
                app.question = "date"
                writeLog(f"예약하고 싶다는 말에 날짜를 물어보기, raw_input: {userText}", 0)
                return "네, 예약을 도와드리겠습니다.<br>어느 날짜(0월 0일)로 예약하실건가요?"
            
            writeLog(f"슬롯이 빈 상태에서 주어진 알 수 없는 입력, raw_input: {userText}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}", 0)
            return '죄송합니다 제가 이해를 잘 못해서 다시 한번 입력해주세요.' + response

        else:
            # 슬롯이 채워지지 않은 것이 있다면 질문 던지기
            writeLog(f"슬롯이 덜 채워져서 질문 던짐, raw_input: {userText}, slot_dict: {app.slot_dict}, token_list: {token_list}, inferred_tags: {inferred_tags}, slots_score: {slots_score}", 0)
            if app.slot_dict['date'] == '':
                app.question = "date"
                return str(np.random.choice(answer_date_arr, 1)[0])+ response
            elif (app.slot_dict['start'] == '') and (app.slot_dict['end'] == ''):
                app.question = "start"
                return str(np.random.choice(answer_start_end_arr, 1)[0]) + response
            elif app.slot_dict['start'] == '':
                app.question = "start"
                return str(np.random.choice(answer_start_arr, 1)[0]) + response
            elif app.slot_dict['end'] == '':
                app.question = "end"
                return str(np.random.choice(answer_end_arr, 1)[0]) + response
            elif app.slot_dict['person'] == '':
                app.question = "person"
                return str(np.random.choice(answer_person_arr, 1)[0]) + response
            elif app.slot_dict['name'] == '':
                app.question = "name"
                return str(np.random.choice(answer_name_arr, 1)[0]) + response
            elif app.slot_dict['phone'] == '':
                app.question = "phone"
                return str(np.random.choice(answer_phone_arr, 1)[0]) + response

    except AttributeError as AE: #세션이 만료됐다면
        writeLog("AttributeError: " + str(AE), 1)
        return "F5를 눌러 페이지를 새로고침 해주세요"

    except Exception as e:
        #그 외 예외 로깅
        exc_type, exc_value, exc_traceback = sys.exc_info()
        writeLog(str(e), 2)
        for tb in traceback.format_exception(exc_type, exc_value, exc_traceback):
            writeLog(tb.strip(), 3)
        
        if app.debug: 
            return str(e)
        else: 
            return "<br>오류가 발생했습니다, 페이지를 다시 열어주세요"
    
    return "이 문장은 출력될 일이 없습니다."


