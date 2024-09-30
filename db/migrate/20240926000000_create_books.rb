class CreateBooks < ActiveRecord::Migration[7.1]
  def change
    create_table :books do |t|
      t.string :title
      t.text :description
      t.belongs_to :category, null: false, foreign_key: true
      t.string :publisher
      t.date :publish_date
      t.integer :price

      t.timestamps
    end

    add_index :books, :title, unique: true
  end
end
