from flask import Flask, request

app = Flask(__name__)

@app.route('/api', methods=['GET'])

def api():
  result = '这是返回的文本数据'
  link = 'https://fanyi.baidu.com/'  # 要返回的链接
  response = {
      'result': result,
      'link': link
  }
  return response

if __name__ == '__main__':
  app.run(host='10.0.4.8', port='7777')