from flask import Flask,request,jsonify
from flask_cors import cross_origin
import pymongo
import os

app = Flask(__name__)
image_list=os.listdir('data/image_data')

def connectDB():
    '''
    mongodb数据库连接
    params: None
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
        success-> True
        fail-> False
    '''
    try:
        result=collection.insert_one(data)#插入数据到集合
        print('####>insert operation is successful')
        return True
    except Exception as e:
        print('####>insert operation is feiled')
        return False

def retrieve(collection,constraint={},order={}):
    '''
    查询数据库中的数据
    params:
        collection: 查询集合对象
        constraint: 查询约束(默认没有约束，查询全部)
        order: 结果排列顺序(默认升序,降序-1)
    returns:
        success-> True,list(dict<result1>,dict<result2>) or []
        fail-> False,error
    '''
    try:
        result=None
        ans=[]
        if order=={}: # 查询结果默认按照_id升序排列
            result=collection.find(constraint).sort("_id",1)
        else:
            result=collection.find(constraint).sort(list(order.keys())[0],list(order.values())[0])
        if result!= None: # 查询不为空
            for item in result:
                item['_id']=str(item['_id']) # 处理_id为字符串，不然不能json化
                item['images']=generateImageLink(item['attributes']['name']) # 获取相关图片的url
                ans.append(item)
        print('####>retrieve operation is successful')
        print('####>retrieve result:',ans)
        return True,ans
    except Exception as e:
        print('!!!!>ERROR: the retrieve operation occurs error ({})'.format(e))
        return False,result

def verifyUser(password):
    '''
    验证用户身份
    params:
        password: 密码
    returns:
        True: 用户验证通过
        False: 用户验证失败
    '''
    right_password='123456'
    if password==right_password:
        return True
    return False

def generateImageLink(name):
    '''
    生成相关图片连接
    params:
        name: 自然保护区名称
    returns:
        [link]: 相关图片url列表
    '''
    links=[]
    for image in image_list:
        if name in image:
            link='http://124.222.93.155:8080/poi_images/{}'.format(image)
            links.append(link)
    return links
    
@app.route('/queryByID',methods=['GET'])
@cross_origin()#解决跨域请求
def queryByID():
    '''
    按id查询
    params:
        password: 权限密码
        id: POI的id
    returns:
        poi
    '''
    # 获取参数
    password=request.args.get('password')
    id=request.args.get('id')
    if verifyUser(password): # 验证权限
        client,db,collection=connectDB() # 连接数据库
        state,ans=retrieve(collection,{'attributes.code':id}) # 查询
        closeDB(client) # 关闭数据库连接
        if state:
            return ans
        else:
            return 'Query by id is failed'
    else:
        return 'You don\'t have access to this api'

@app.route('/queryByProvince',methods=['GET'])
@cross_origin()#解决跨域请求
def queryByProvince():
    '''
    按省份查询
    params:
        password: 权限密码
        province: POI的省份
        order: 1(default,按照FID升序);-1(按照FID降序)
    returns:
        poi
    '''
    # 获取参数
    password=request.args.get('password')
    province=request.args.get('province')
    order=int(request.args.get('order'))
    if verifyUser(password): # 验证权限
        client,db,collection=connectDB() # 连接数据库
        state,ans=retrieve(collection,{'attributes.Province':province},{'attributes.FID':order}) # 查询
        closeDB(client) # 关闭数据库连接
        if state:
            return ans
        else:
            return 'Query by province is failed'
    else:
        return 'You don\'t have access to this api'

@app.route('/queryByRectangle',methods=['GET'])
@cross_origin()#解决跨域请求
def queryByRectangle():
    '''
    按矩阵范围查询
    params:
        password: 权限密码
        minLon: 最小经度
        maxLon: 最大经度
        minLat: 最小纬度
        maxLat: 最大纬度
        order: 1(default,按照FID升序);-1(按照FID降序)
    returns:
        poi
    '''
    # 获取参数
    password=request.args.get('password')
    minLon=float(request.args.get('minLon'))
    maxLon=float(request.args.get('maxLon'))
    minLat=float(request.args.get('minLat'))
    maxLat=float(request.args.get('maxLat'))
    order=int(request.args.get('order'))
    if verifyUser(password): # 用户权限认证
        # 构造查询矩阵
        rectangleCoordinates = [(minLon, minLat), (minLon, maxLat), (maxLon, maxLat), (maxLon, minLat), (minLon, minLat)]
        print(rectangleCoordinates)
        client,db,collection=connectDB()
        query = { # 查询条件
            "geometry": {
                "$geoWithin":{
                    "$geometry":{
                    "type": "Polygon",
                    "coordinates": [rectangleCoordinates]
                    }
                }
            }
        }
        state,ans=retrieve(collection,query,{'attributes.FID':order})
        closeDB(client)
        if state:
            return ans
        else:
            return "Query by rectangle failed"
    else:
        return 'You don\'t have access to this api'

@app.route('/queryByCircle',methods=['GET'])
@cross_origin()#解决跨域请求
def queryByCircle():
    '''
    按矩阵范围查询
    params:
        password: 权限密码
        centerLon: 中心点经度
        centerLat: 中心点纬度
        radiusByKm: 半径km
        order: 1(default,按照FID升序);-1(按照FID降序)
    returns:
        poi
    '''
    # 获取参数
    password=request.args.get('password')
    centerLon=float(request.args.get('centerLon'))
    centerLat=float(request.args.get('centerLat'))
    radiusByKm=float(request.args.get('radiusByKm'))
    order=int(request.args.get('order'))
    if verifyUser(password): # 用户权限认证
        radiusByRadians = radiusByKm / 6371 # 将半径转换为弧度，因为之后构造的圆形区域半径参数需要是弧度制(地球半径是6371km)
        query = { # 查询条件
            "geometry": {
                "$geoWithin":{
                    "$centerSphere": [(centerLon, centerLat), radiusByRadians]
                }
            }
        }
        client,db,collection=connectDB()
        state,ans=retrieve(collection,query,{'attributes.FID':order})
        closeDB(client)
        if state:
            return ans
        else:
            return "Query by circle failed"
    else:
        return 'You don\'t have access to this api'


if __name__ == '__main__':

    app.run(host='10.0.4.8',debug=True,port='7777')

