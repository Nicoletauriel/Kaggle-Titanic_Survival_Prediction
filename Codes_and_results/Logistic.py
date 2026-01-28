import pandas as pd  #引入pandas库
train = pd.read_csv('../Data/train.csv')  #读取数据并且将其命名为变量train，两个句点表示回到代码所在文件夹上一级检索数据
train.head()  #print出前五行数据，展示数据情况，全部print出来可能会卡死

print("缺失值统计：")
print(train.isnull().sum())  #查看缺失值

print("\n统计摘要：")  #\n代表换行
print(train.describe())  #查看摘要

print("不同性别的平均生还率：")
print(train.groupby('Sex')['Survived'].mean())  #groupby是pandas库中的一个分组函数，此处用于按照sex来分组，计算survived的均值

print("\n性别 + 舱位等级的综合生还率：")
pivot_result = train.pivot_table(index='Sex', columns='Pclass', values='Survived', aggfunc='mean')  #看sex和pclass两个因素同时作用下的生还率
print(pivot_result)

median_ages = train.groupby(['Sex', 'Pclass'])['Age'].transform('median') # 按 Sex 和 Pclass 分组，计算 Age 的中位数

train['Age'] = train['Age'].fillna(median_ages) # fillna：如果是缺失值（NaN），就用括号里的值填进去

print("填补后的缺失值数量：", train['Age'].isnull().sum())

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1}) # 使用 map 函数进行“翻译”

print("看看性别的操作是否成功：")
print(train['Sex'].head())

train['Embarked'] = train['Embarked'].fillna('S') #用人数最多的港口S填补港口缺失值
train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) #继续使用map函数翻译

print("\n看看港口的操作是否成功")
print(train['Embarked'].head())

from sklearn.linear_model import LogisticRegression

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'] #选择因果变量
X = train[features]
y = train['Survived'] # 想要预测的目标

model = LogisticRegression(max_iter=1000) #模型初始化

model.fit(X, y) #开始训练

score = model.score(X, y)
print(f"模型在训练集上的准确率: {score:.2%}")

test = pd.read_csv('../Data/test.csv')  #读取数据并且将其命名为变量test，两个句点表示回到代码所在文件夹上一级检索数据
test.head()  #print出前五行数据，展示数据情况，全部print出来可能会卡死

print("缺失值统计：")
print(test.isnull().sum())  #查看缺失值

print("\n统计摘要：")  #\n代表换行
print(test.describe())  #查看摘要

median_ages = test.groupby(['Sex', 'Pclass'])['Age'].transform('median') # 按 Sex 和 Pclass 分组，计算 Age 的中位数

test['Age'] = test['Age'].fillna(median_ages) # fillna：如果是缺失值（NaN），就用括号里的值填进去

test['Fare'] = test['Fare'].fillna(test['Fare'].median())

print("填补后的缺失值数量：", test['Age'].isnull().sum())

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}) # 使用 map 函数进行“翻译”

print("看看性别的操作是否成功：")
print(test['Sex'].head())

test['Embarked'] = test['Embarked'].fillna('S') #用人数最多的港口S填补港口缺失值
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}) #继续使用map函数翻译

print("\n看看港口的操作是否成功")
print(test['Embarked'].head())

X_test = test[features] #给预测选自变量

test_predictions = model.predict(X_test) #做出预测

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_predictions
})

submission.to_csv('submission_final_logistics.csv', index=False)
print("Success")


