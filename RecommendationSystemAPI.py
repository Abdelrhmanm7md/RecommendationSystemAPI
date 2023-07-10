from flask import Flask ,request, jsonify, make_response
from flask_restful import abort
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS

app = Flask(__name__ )
CORS(app)

@app.route('/recommend/<int:id>', methods = ['GET'])
def get(id):

    x = movie_recommend(id)
    xx = list(x)
    return make_response(jsonify(xx), 200)



def movie_recommend(id_movie):
        df = pd.read_csv('dataset.csv')

        df = df[['title','overview','id']]

        df.dropna(inplace=True)

        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0, stop_words='english')

        matrix = tf.fit_transform(df['overview'])

        cosine_similarities = linear_kernel(matrix,matrix)

        movie_id = df['id']

        indices = pd.Series(df.index, index=df['id'])

        idx = indices[id_movie]

        sim_scores = list(enumerate(cosine_similarities[idx]))

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1:33]

        movie_indices = [i[0] for i in sim_scores]

        x = movie_id.iloc[movie_indices]
        #return series
        return x.head(10)











if __name__ == "__main__"    :
    app.run(debug=True)
