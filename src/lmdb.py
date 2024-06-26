# -*- coding: utf-8 -*-
import lmdb

# 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
# map_size定义最大储存容量，单位是byte，以下定义1TB容量
# env = lmdb.open('./train', map_size=1099511627776)
# env.close()
# # -*- coding: utf-8 -*-
#
# # map_size定义最大储存容量，单位是kb，以下定义1TB容量
# env = lmdb.open("./train", map_size=1099511627776)
#
# txn = env.begin(write=True)
#
# # 添加数据和键值
# txn.put(key='1', value='aaa')
# txn.put(key='2', value='bbb')
# txn.put(key='3', value='ccc')
#
# # 通过键值删除数据
# txn.delete(key='1')
#
# # 修改数据
# txn.put(key='3', value='ddd')
#
# # 通过commit()函数提交更改
# txn.commit()
# env.close()

if __name__ == '__main__':
    print('{0:<8}{0:<8}'.format('Data','Time'))  # 占5个字符空间，0是format参数中的变量索引
    print('{0:<5}'.format('123'))  # 占5个字符空间，0是format参数中的变量索引
    print('{0:<5}'.format('00123'))  # 占5个字符空间，0是format参数中的变量索引
