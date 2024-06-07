from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Model ve gerekli verileri yükleme
model = joblib.load('nmf_model_2.pkl')
interests_index_mentee = joblib.load('interests_index_mentee.pkl')
interests_index_mentor = joblib.load('interests_index_mentor.pkl')

# Eğitim verilerini yükleme (mentor bilgileri için)
train_df = pd.read_csv('train_df.csv')

# Mentee ve mentor ilgi alanlarını kodlama fonksiyonu
def encode_interests(interest_list, interests_index):
    encoded = np.zeros(len(interests_index))
    for interest in interest_list.split(", "):
        if interest in interests_index:
            encoded[interests_index[interest]] = 1
    return encoded

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    mentee_interests = data['mentee_interests']
    mentee_category = data['mentee_category']
    
    # Mentee ilgi alanlarını kodlama
    encoded_mentee_interests = encode_interests(mentee_interests, interests_index_mentee)
    
    # Mentee faktörlerini model ile elde etme
    mentee_factors = model.transform(np.array([encoded_mentee_interests]))
    
    # Benzerlik hesaplamaları 
    mentor_factors_train = model.transform(np.array(train_df['encoded_mentor_interests'].tolist()))
    similarities = np.dot(mentee_factors, mentor_factors_train.T).flatten()
    
    # Her mentee için en uygun N mentoru önerme
    N = 10  # Her mentee için önerilen mentor sayısı
    recommendations = []

    # Benzerlik puanlarına göre sıralı indeksler
    mentor_indices = np.argsort(similarities)[::-1]

    # Mentee'nin kategorisini alma
    recommended_mentors = set()  # Her bir mentee için önerilen mentorların bir kümesi
    
    # Mentorları benzerlik puanına göre sırayla kontrol et
    for mentor_index in mentor_indices:
        # Eğer önerilen mentor sayısı N'e ulaştıysa döngüyü sonlandır
        if len(recommended_mentors) >= N:
            break

        # Mentorun kategorisini ve ID'sini al
        mentor_category = train_df.iloc[mentor_index]['Mentor_Category']
        mentor_id = train_df.iloc[mentor_index]['Mentor_ID']
        mentor_name = train_df.iloc[mentor_index]['Mentor_Name']
        mentor_surname = train_df.iloc[mentor_index]['Mentor_Surname']

        # Eğer mentorun kategorisi mentee'nin kategorisine eşitse ve daha önce öneri yapılmadıysa
        if mentor_category == mentee_category and mentor_id not in recommended_mentors:
            recommendations.append({
                'mentee_id': data['mentee_id'],
                'mentor_id': mentor_id,
                'mentor_name': mentor_name,
                'mentor_surname': mentor_surname
            })
            recommended_mentors.add(mentor_id)

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
