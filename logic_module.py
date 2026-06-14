import sqlite3
from contextlib import contextmanager
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "my_wardrobe.db"

TOPS_LIST = ['Tshirts', 'Shirts', 'Top', 'Tops', 'Sweaters', 'Jackets']
BOTTOMS_LIST = ['Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants']
SHOES_LIST = ['Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats']
DRESS_LIST = ['Dresses']


@contextmanager
def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wardrobe (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                style TEXT NOT NULL
            )
            """
        )


def add_wardrobe_item(image_path, category, style):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO wardrobe (image_path, category, style)
            VALUES (?, ?, ?)
            """,
            (image_path, category, style),
        )
        return cursor.lastrowid


def get_wardrobe_items():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, image_path, category, style
            FROM wardrobe
            ORDER BY id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_random_item(connection, target_style, categories):
    placeholders = ", ".join("?" for _ in categories)
    row = connection.execute(
        f"""
        SELECT image_path
        FROM wardrobe
        WHERE style = ? AND category IN ({placeholders})
        ORDER BY RANDOM()
        LIMIT 1
        """,
        (target_style, *categories),
    ).fetchone()
    return row["image_path"] if row else None


def suggest_outfit(user_chosen_item_path):
    with get_connection() as connection:
        chosen_item = connection.execute(
            """
            SELECT image_path, category, style
            FROM wardrobe
            WHERE image_path = ?
            LIMIT 1
            """,
            (user_chosen_item_path,),
        ).fetchone()

        if chosen_item is None:
            return {'Top': None, 'Bottom': None, 'Shoes': None}, None

        target_style = chosen_item["style"]
        target_cat = chosen_item["category"]
        outfit = {'Top': None, 'Bottom': None, 'Shoes': None}

        if target_cat in TOPS_LIST:
            outfit['Top'] = user_chosen_item_path
            outfit['Bottom'] = get_random_item(
                connection, target_style, BOTTOMS_LIST
            )
            outfit['Shoes'] = get_random_item(
                connection, target_style, SHOES_LIST
            )
        elif target_cat in DRESS_LIST:
            outfit['Top'] = user_chosen_item_path
            outfit['Shoes'] = get_random_item(
                connection, target_style, SHOES_LIST
            )
        elif target_cat in BOTTOMS_LIST:
            outfit['Bottom'] = user_chosen_item_path
            outfit['Top'] = get_random_item(
                connection, target_style, TOPS_LIST
            )
            outfit['Shoes'] = get_random_item(
                connection, target_style, SHOES_LIST
            )
        elif target_cat in SHOES_LIST:
            outfit['Shoes'] = user_chosen_item_path
            outfit['Top'] = get_random_item(
                connection, target_style, TOPS_LIST
            )
            outfit['Bottom'] = get_random_item(
                connection, target_style, BOTTOMS_LIST
            )
        else:
            outfit['Top'] = user_chosen_item_path

    return outfit, target_style
