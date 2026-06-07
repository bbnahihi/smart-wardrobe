import pandas as pd


DB_PATH = "my_wardrobe_db.csv"

TOPS_LIST = ['Tshirts', 'Shirts', 'Top', 'Tops', 'Sweaters', 'Jackets']
BOTTOMS_LIST = ['Jeans', 'Trousers', 'Shorts', 'Skirts', 'Track Pants']
SHOES_LIST = ['Casual Shoes', 'Formal Shoes', 'Sports Shoes', 'Heels', 'Flats']
DRESS_LIST = ['Dresses']


def suggest_outfit(user_chosen_item_path):
    wardrobe_df = pd.read_csv(DB_PATH)

    chosen_rows = wardrobe_df[wardrobe_df['image_path'] == user_chosen_item_path]
    if chosen_rows.empty:
        return {'Top': None, 'Bottom': None, 'Shoes': None}, None

    chosen_item = chosen_rows.iloc[0]
    target_style = chosen_item['style']
    target_cat = chosen_item['category']

    matching_items = wardrobe_df[wardrobe_df['style'] == target_style]
    outfit = {'Top': None, 'Bottom': None, 'Shoes': None}

    def get_random_item(cat_list):
        subset = matching_items[matching_items['category'].isin(cat_list)]
        if not subset.empty:
            return subset.sample(1)['image_path'].values[0]
        return None

    if target_cat in TOPS_LIST:
        outfit['Top'] = user_chosen_item_path
        outfit['Bottom'] = get_random_item(BOTTOMS_LIST)
        outfit['Shoes'] = get_random_item(SHOES_LIST)

    elif target_cat in DRESS_LIST:
        outfit['Top'] = user_chosen_item_path
        outfit['Shoes'] = get_random_item(SHOES_LIST)

    elif target_cat in BOTTOMS_LIST:
        outfit['Bottom'] = user_chosen_item_path
        outfit['Top'] = get_random_item(TOPS_LIST)
        outfit['Shoes'] = get_random_item(SHOES_LIST)

    elif target_cat in SHOES_LIST:
        outfit['Shoes'] = user_chosen_item_path
        outfit['Top'] = get_random_item(TOPS_LIST)
        outfit['Bottom'] = get_random_item(BOTTOMS_LIST)

    else:
        outfit['Top'] = user_chosen_item_path

    return outfit, target_style
