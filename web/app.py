from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import rec_sys_model

app = Flask(__name__)

# Чтение данных из CSV при старте приложения
df = pd.read_csv('../Datasets/BooksDatasetClean.csv')

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

    return render_template('index.html', books=books, page=page, total_pages=total_pages)

# Детальная страница книги
@app.route('/book/<title>', methods=['GET'])
def book_details(title):
    # Ищем книгу по названию
    book = df[df['Title'] == title].to_dict(orient='records')
    if not book:
        return "Book not found", 404

    book = book[0]  # Берём первую (и единственную) найденную запись
    return render_template('book_details.html', book=book)

if __name__ == '__main__':
    app.run(debug=True)
