# -*- coding: utf-8 -*-

from app.main import app
import webbrowser

if __name__ == "__main__":
    app.run("localhost",port=5000, debug=True) # 테스트할때는 이것으로 쓰세요
    #app.run() #ngrok 쓸 때는 이거로 쓰세요
    webbrowser.open("http://localhost:5000", 2)