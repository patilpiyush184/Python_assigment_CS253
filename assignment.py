from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import pandas as pd

# Step 1: Data Preprocessing
data = pd.read_csv('train.csv')
data1 = pd.read_csv('test.csv')
data.drop(columns=['ID','Candidate','Constituency ∇'], inplace=True)
data1.drop(columns=['ID','Candidate','Constituency ∇'], inplace=True)

def convert_to_lakhs(value):
    value = str(value)
    if value == "0":
        return 0
    elif 'Crore+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number * 100)
    elif 'Lac+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number)
    elif 'Thou+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number / 100)
    elif 'Hund+' in value:
        number, unit = value.split()
        number = float(number.replace(",", ""))
        return int(number / 1000)
    else:
        return int(value)

data["Total Assets"] = data["Total Assets"].apply(convert_to_lakhs)
data["Liabilities"] = data["Liabilities"].apply(convert_to_lakhs)
data1["Total Assets"] = data1["Total Assets"].apply(convert_to_lakhs)
data1["Liabilities"] = data1["Liabilities"].apply(convert_to_lakhs)

label_encoder = LabelEncoder()
data['Education'] = label_encoder.fit_transform(data['Education'])

data = pd.get_dummies(data, columns=['Party','state'])
data1 = pd.get_dummies(data1, columns=['Party','state'])

X = data.drop(columns=['Education'])
y = data['Education']

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_test_scaled = scaler.transform(X_test_split)
data1_scaled = scaler.transform(data1)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train_split)

y_pred = svm_model.predict(data1_scaled)

y_pred = label_encoder.inverse_transform(y_pred)
df_pred = pd.DataFrame(y_pred, columns=["Education"])
df_pred.to_csv('submission.csv', index_label='ID')
