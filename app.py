from flask import Flask, render_template, request
from recommender import recommend_by_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_summary = request.form['movie_summary']
    recommendations = recommend_by_summary(user_summary, num_recommendations=5)

    if not recommendations.empty:
        return render_template(
            'index.html',
            user_summary=user_summary,
            recommendations=recommendations.to_dict(orient='records')
        )
    else:
        return render_template('index.html', error="No similar movies found!", user_summary=user_summary)

if __name__ == '__main__':
    app.run(debug=True)
