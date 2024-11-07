from flask import Flask, request, jsonify, render_template, url_for, redirect
import pandas as pd
from rec_sys_model.rec_sys_model import recSysModel

app = Flask(__name__)

# Чтение данных из CSV при старте приложения
df = pd.read_csv('../Datasets/BooksDatasetClean.csv')
metric = pd.read_csv('../Datasets/rating.csv')
recsys = recSysModel('../Datasets/BooksDatasetClean.csv')
recsys.load('../Datasets/new_df.csv')
embeddings_df = pd.read_csv('../Datasets/new_df.csv')


# Главная страница с пагинацией
@app.route('/', methods=['GET'])
@app.route('/page/<int:page>', methods=['GET'])
def home(page=1):
    per_page = 10  # Количество книг на одной странице
    total_pages = (len(df) // per_page) + 1  # Всего страниц
    start = (page - 1) * per_page
    end = start + per_page

    # Выбираем нужные книги для текущей страницы
    books = df[start:end].to_dict(orient='records')

    # Генерируем URL для каждой книги
    for book in books:
        book['details_url'] = url_for('book_details', title=book['Title'])
        book['metric_url'] = url_for('book_metric', title=book['Title'])

    return render_template('index.html', books=books, page=page, total_pages=total_pages)

# Детальная страница книги
@app.route('/book/<title>', methods=['GET'])
def book_details(title):
    # Ищем книгу по названию
    book = df[df['Title'] == title].to_dict(orient='records')
    record = embeddings_df[embeddings_df['name'] == title]

    if not book:
        return "Book not found", 404

    book = book[0]  # Берём первую найденную запись
    recommended_books = recsys.predict(record, n=10)
    recommended_books_links = [
        {"name": rec_book, "url": url_for('book_details', title=rec_book)}
        for rec_book in recommended_books
    ]

    return render_template('book_details.html', book=book, recommended_books=recommended_books_links)

@app.route('/metric/<title>', methods=['GET'])
def book_metric(title):
    book = df[df['Title'] == title].to_dict(orient='records')
    record = embeddings_df[embeddings_df['name'] == title]

    if not book:
        return "Book not found", 404

    book = book[0]  # Берём первую (и единственную) найденную запись
    recommended_books = recsys.predict(record, n=100)
    recommended_books = [
        {"name": rec_book, "url": url_for('book_details', title=rec_book),
         "description":  df[df['Title'] == rec_book]['Description'].to_string(index=False)}
        for rec_book in recommended_books
    ]

    return render_template('book_metric.html', book=book, recommended_books=recommended_books)

@app.route('/rate', methods=['POST'])
def rate_book():
    global metric
    rating = int(request.form.get('rating'))
    source_book = request.form.get('source_book')
    recommended_book = request.form.get('recommended_book')

    new_row = pd.DataFrame([{
        'Title': source_book,
        'Title_for_rate': recommended_book,
        'rate': rating
    }])

    # Объединение новой строки с существующим DataFrame
    metric = pd.concat([metric, new_row], ignore_index=True)

    # Проверка содержимого DataFrame перед сохранением
    print(metric)

    # Сохранение изменений в CSV
    metric.to_csv('../Datasets/rating.csv', index=False)
    return jsonify({"message": "Rating added successfully"})

from flask import request, render_template, url_for

@app.route('/search', methods=['GET'])
def search():
    title = request.args.get('title')  # Получаем параметр title из строки запроса
    recommended_books = recsys.closest_title(title, 10)
    recommended_books_links = [ df[df['Title'] == title] for title in recommended_books ]
    return render_template('closest_titles.html', recommended_books=recommended_books_links)

if __name__ == '__main__':
    # app.run(host='192.168.0.105')
    app.run(debug=True)
    