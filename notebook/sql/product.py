import json

import pandas as pd
from typing import List, Optional

from sqlalchemy import create_engine, Column, String, Table, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Product(Base):
    __tablename__ = 'product'
    product_id = Column(String, primary_key=True)
    product_name = Column(String)
    product_code = Column(String)
    color = Column(String)
    size = Column(String)
    note = Column(String)

    def to_dict(self):
        return {
            'product_id': self.product_id,
            'product_name': self.product_name,
            "product_code": self.product_code,
            'color': self.color,
            'size': self.size,
            'note': self.note
        }

engine = create_engine('sqlite:///cache/product.db')
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

df_products = pd.read_excel('data/gam_chat_clean_gpt.xlsx')
df_products['product_preprocess'] = [json.loads(x) for x in df_products['product_preprocess'].tolist()]

products = []
for i, row in df_products.iterrows():
    products.append(row.to_dict())

from sqlalchemy.exc import IntegrityError

for p in products:
    if len(p['product_preprocess']) > 1: continue  # Temporary skip combo products

    product_preprocess = p['product_preprocess'][0]
    product = Product(
        product_id=str(p['product_id']),
        product_name=p['product_name'],
        product_code=product_preprocess['type'],
        color=product_preprocess['color'],
        size=product_preprocess['size'],
        note=product_preprocess['note']
    )
    try:
        session.add(product)
        session.commit()
    except IntegrityError:
        session.rollback()
        print(f"Product with ID {p['product_id']} already exists.")
        print(f"Stop Insert.")
        break

def find_db_products_by_properties(product_code: str, color: str, size: str) -> List[Product]:
    if color is None and size is None and product_code is None:
        raise ValueError("At least one of 'color', 'size' or 'product_code' must be specified")

    query = session.query(Product)
    
    if color is not None:
        query = query.filter(Product.color.ilike(f"%{color}%"))
    if product_code is not None and product_code != "":
        query = query.filter(Product.product_code == product_code)
    if size is not None and size != "":
        query = query.filter(Product.size == size)
    
    products = query.limit(3).all()
    return products

if __name__ == "__main__":
    product_code = "A1"
    color_to_search = 'đỏ'  # Replace with the color you want to search for
    size_to_search = None     # Replace with the size you want to search for

    found_products = find_db_products_by_properties(product_code, color_to_search, size_to_search)

    for product in found_products:
        print(f"Product ID: {product.product_id}, Name: {product.product_name}, Color: {product.color}, Size: {product.size}")