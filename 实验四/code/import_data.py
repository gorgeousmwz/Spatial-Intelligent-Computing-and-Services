import json
import pymongo

def read_data(path):
    '''
    读取文件
    params: 文件路径
    returns: 读取的数据
    '''
    # 读取JSON文件
    with open(path) as json_file:
        data = json.load(json_file)
    return data

def connectDB():
    '''
    mongodb数据库连接
    params: None
    returns:
        returns: 
        client: 连接对象
        db: 数据库对象
        collection: 集合对象
    '''
    client=pymongo.MongoClient(host='localhost',port=27017,username='mwz',password='gorgeous') # 创建连接
    db=client.POI # 指定POI数据库
    collection=db.poi # 指定poi集合
    print('---->mongodb client is established')
    return client,db,collection

def closeDB(client):
    '''
    关闭数据库连接
    params: 连接对象
    returns: None
    '''
    client.close()
    print('---->mongodb client close')

def add(collection,data):
    '''
    向数据库中添加数据
    params: 集合对象,插入数据
    returns: 
        success-> True,
        fail-> False
    '''
    try:
        result=collection.insert_many(data['features'])#插入数据到集合
        return True
    except Exception as e:
        return False

client,db,collection=connectDB()
data=read_data('data/2016自然保护区POI.json')
add(collection,data)
closeDB(client)

