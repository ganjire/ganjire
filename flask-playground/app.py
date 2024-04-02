from flask import Flask, jsonify, render_template, request
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sqlalchemy

app = Flask(__name__)

def fetch_data():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.59 Safari/537.36'
    }
    url = 'https://www.imdb.com/chart/top'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        movies = soup.find_all('li', class_='ipc-metadata-list-summary-item')

        data = []

        for movie in movies:
            title_element = movie.find('h3', class_='ipc-title__text')
            if title_element:
                # Zuerst wird der gesamte Text extrahiert und unnötige Leerzeichen entfernt
                full_title_text = title_element.text.strip()
                
                # Trennen des Textes in Ranking und Titel, basierend auf dem ersten Leerzeichen
                split_text = full_title_text.split(' ', 1) # Begrenzt den Split auf das erste Vorkommen
                if len(split_text) > 1:
                    Ranking, Title = split_text[0], split_text[1]
                else:
                    Ranking = "Unknown"
                    Title = full_title_text
            
                # Extrahieren des Jahres aus dem `span`-Element
                year_element = movie.find('span', class_='cli-title-metadata-item')
                if year_element:
                    Year = year_element.text.strip()
                else:
                    Year = "Jahr unbekannt"
            
                rating_element = movie.find('span', class_='ipc-rating-star--imdb')
                if rating_element:
                    # Extrahiere den gesamten Text, der das Rating und die Anzahl der Stimmen enthält
                    rating_vote_text = rating_element.text.strip()
                    
                    # Trenne das Rating von der Anzahl der Stimmen, basierend auf der Annahme,
                    # dass das Rating am Anfang steht und die Anzahl der Stimmen danach kommt
                    rating_text_parts = rating_vote_text.split('(')
                    Rating = rating_text_parts[0].strip()
                    if len(rating_text_parts) > 1:
                        # Entferne die schließende Klammer und extrahiere nur die Anzahl der Stimmen
                        VoteCount = rating_text_parts[1].replace(')', '').strip()
                    else:
                        VoteCount = "Stimmenanzahl unbekannt"
                else:
                    Rating = "Rating unbekannt"
                    VoteCount = "Stimmenanzahl unbekannt"
                    
                # Füge die extrahierten Daten zur Liste hinzu
                data.append([Ranking, Title, Year, Rating, VoteCount])

        # Erstelle das DataFrame
        df = pd.DataFrame(data, columns=['Ranking', 'Title', 'Year', 'Rating', 'Vote Count'])
        print(df)

        # Funktion zur Konvertierung von Vote Count
        def convert_vote_count(vote_str):
            if 'M' in vote_str:
                return float(vote_str.replace('M', '')) * 1000000
            elif 'K' in vote_str:
                return float(vote_str.replace('K', '')) * 1000
            else:
                return float(vote_str)
            
        # Konvertiere die 'Vote Count'-Daten
        df['Vote Count'] = df['Vote Count'].apply(lambda x: convert_vote_count(x) if x != "Stimmenanzahl unbekannt" else np.nan)
        df['Ranking'] = df['Ranking'].str.replace('.', '').astype(int)
        df['Year'] = df['Year'].astype(float)
        df['Rating'] = df['Rating'].astype(float)

        # Aufteilen der Daten
        X = df[['Year', 'Vote Count', 'Ranking']]
        y = df['Rating']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, df
    
    # Führe eine Datenbankverbindung aus
    # Erstelle eine Verbindung zur MySQL-Datenbank mit sqlalchemy
    database_username = 'root'
    database_password = 'RockyRebi12'
    database_ip       = '127.0.0.1'
    database_name     = 'imdb_data'
    database_connection = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.
                                                format(database_username, database_password, 
                                                        database_ip, database_name))

    # Dein DataFrame
    df = pd.DataFrame(data, columns=['Ranking', 'Title', 'Year', 'Rating', 'Vote Count'])

    # Nutze pandas, um den DataFrame direkt in die Datenbank zu laden
    df.to_sql(con=database_connection, name='movies', if_exists='replace', index=False)

    # Schließe die Verbindung
    database_connection.dispose()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_data')
def get_data():
    X_train, X_test, y_train, y_test, df = fetch_data()
    return jsonify({'train_data': X_train.to_dict(), 'test_data': X_test.to_dict(), 'train_labels': y_train.tolist(), 'test_labels': y_test.tolist(), 'dataframe': df.to_dict()})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    year = data['year']
    ranking = data['ranking']
    vote_count = data['vote_count']
    
    X_train, X_test, y_train, y_test, _ = fetch_data()
    
    # Modell trainieren
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Vorhersage und Bewertung
    prediction = model.predict([[year, vote_count, ranking]])
    
    # Parameter anzeigen
    print(model.get_params())

    # Wichtigkeit jedes Features 
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importances = pd.Series(importances, index=feature_names)

    print(feature_importances.sort_values(ascending=False))

    
    #return jsonify({'rating_prediction': prediction[0]})
    return jsonify({'rating_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
