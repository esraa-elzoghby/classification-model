# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#بنجيب ال library الهنستخدمها في المشروع كله 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # بجيب كلاس Decision Tree Classifier
from sklearn.model_selection import train_test_split # بجيب كلاس train_test_split function
from sklearn import metrics #كلاس scikit-learn metrics module for accuracy calculation
from sklearn.impute import SimpleImputer #الكلاس الهعمل بيه clean لل data
import numpy as np 
import pickle 
#بحدد ال cols بتاعت الجدول علشان لو حبيت اعرضها هنا تتعرض بشكل كويس وابقا عارف ايه المعروض قدامي
col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak','slope','ca','thal','num']
# بجيب الملف النا محمله من علي النت بتاع ال dataset
pima = pd.read_csv("heart_disease_dataset.csv", header=None, names=col_names)
pima.head()
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang','oldpeak','slope','ca','thal']
#المعطيات
inputs = pima[feature_cols]
#الناتح في الجدول
results = pima.num
#هعوض عن البيانات الفيها المشكلة
#باخد object من الكلاس 
imputed_Model = SimpleImputer(missing_values = -100000,strategy='median')#بعمل clean لل data
# هنخزن البيانت الجديه في inputs 
imputed_inserts = imputed_Model.fit(inputs)
inputs = imputed_inserts.transform(inputs)
#كده انا حليت المشكلة الممكن تكون عندي في البيانات بتاعت ال dataset
#هقسم الداتا هنا علشان ادي جزء لل model وال جزء الفاضل هعمل بيه اختبار لل model علشان اعرف كفاءتها
X_train, X_test, y_train, y_test = train_test_split(inputs, results, test_size=0.3, random_state=1) #30% لأختبار and 70% ال هديه لل model
# دلوقتي هبداء ابني ال descion tree بتاعتي
# 1- خدت object من الكلاس 
clf = DecisionTreeClassifier(criterion="entropy")
# هدي ال descion tree القيم الهنديها لل model 70% 
clf = clf.fit(X_train,y_train)
#هعمل رن لل 30% واسجل النتيجة بتاعتهم

pickle.dump(clf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

y_pred = model.predict(X_test)
# هطبع نسبة الكفاءة لمجرد الحنيكة فقط 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#بعمل check لمريض واحد
h_pred = clf.predict(np.array([[67,1,4,120,229,0,2,129,1,2.6,2,2,7]]))
#0=HEART DISEASE  1=NO HEART DISEASE
print(h_pred)