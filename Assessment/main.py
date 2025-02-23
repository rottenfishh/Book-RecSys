from book_embeddings_model import embeddings_model

model_path = "../models/books_embeddings_new_dataset.npy"

model = embeddings_model.recSysModel(model_path)



def compare_lists(list1, list2):
    list1 = list(map(str.lower, list1))
    list2 = list(map(lambda x: str.lower(x[0]), list2))
    intersection = set(list1) & set(list2)  
    count = len(intersection)

    positions = []
    for item in intersection:
        pos1 = [i for i, x in enumerate(list1) if x == item]
        pos2 = [i for i, x in enumerate(list2) if x == item]
        positions.append((item, pos1, pos2))

    return {
        "intersect_count": count,
        "intersecting_items": positions
    }

recs = {
    "King Lear": [
        "Hamlet", "Macbeth", "Othello", "Romeo and Juliet", "Julius Caesar",
        "Richard III", "The Tempest", "Much Ado About Nothing", "Twelfth Night", "A Midsummer Night's Dream",
        "The Merchant of Venice", "Henry V", "Coriolanus", "Timon of Athens", "Measure for Measure",
        "Antony and Cleopatra", "Cymbeline", "Pericles", "The Winter's Tale", "As You Like It",
        "The Duchess of Malfi", "Doctor Faustus", "Paradise Lost", "The Spanish Tragedy", "Volpone",
        "The Alchemist", "Every Man in His Humour", "Tamburlaine", "Edward II", "The Revenger’s Tragedy"
    ],
    "Anna Karenina": [
        "War and Peace", "Crime and Punishment", "The Brothers Karamazov", "The Idiot", "Resurrection",
        "Fathers and Sons", "The Master and Margarita", "Dead Souls", "The Death of Ivan Ilyich", "Eugene Onegin",
        "Doctor Zhivago", "We", "The Overcoat", "A Hero of Our Time", "Notes from Underground",
        "The Cherry Orchard", "The Lower Depths", "Oblomov", "Hadji Murat", "The Kreutzer Sonata",
        "The Gambler", "White Nights", "Demons", "Nikolai Gogol's Collected Stories", "The Seagull",
        "Three Sisters", "Uncle Vanya", "The Steppe", "The Storm", "The Cossacks"
    ],
    "1984": [
        "We", "Brave New World", "Fahrenheit 451", "Animal Farm", "The Handmaid's Tale",
        "A Clockwork Orange", "The Man in the High Castle", "Do Androids Dream of Electric Sheep?", "The Giver", "Neuromancer",
        "Snow Crash", "The Hunger Games", "The Road", "Blindness", "Oryx and Crake",
        "Never Let Me Go", "The Left Hand of Darkness", "The Dispossessed", "Foundation", "I, Robot",
        "Dune", "The Stars My Destination", "Ubik", "Hyperion", "The Lathe of Heaven",
        "The Iron Heel", "The Parable of the Sower", "Children of Men", "The Minority Report", "Player Piano"
    ],
    "Green Book": [
        "To Kill a Mockingbird", "The Help", "I Know Why the Caged Bird Sings", "Beloved", "The Color Purple",
        "A Lesson Before Dying", "Native Son", "Go Tell It on the Mountain", "The Underground Railroad", "Between the World and Me",
        "The Warmth of Other Suns", "The Nickel Boys", "Invisible Man", "The Fire Next Time", "Their Eyes Were Watching God",
        "A Raisin in the Sun", "Song of Solomon", "The Hate U Give", "Passing", "Cane",
        "Black Boy", "Just Mercy", "The New Jim Crow", "The Souls of Black Folk", "Stamped from the Beginning",
        "March", "Born a Crime", "Americanah", "The Vanishing Half", "There There"
    ],
    "Moby-Dick": [
        "Heart of Darkness", "The Old Man and the Sea", "Treasure Island", "The Odyssey", "Robinson Crusoe",
        "Twenty Thousand Leagues Under the Sea", "The Sea-Wolf", "Billy Budd, Sailor", "The Narrative of Arthur Gordon Pym", "Lord Jim",
        "Kidnapped", "Typee", "White Jacket", "Bartleby, the Scrivener", "The Call of the Wild",
        "The Scarlet Letter", "Gulliver's Travels", "Captain Blood", "The Tempest", "The Last of the Mohicans",
        "The Red Badge of Courage", "The Heart of the Sea", "Rime of the Ancient Mariner", "Beowulf", "Frankenstein",
        "The Count of Monte Cristo", "The Brothers Karamazov", "Blood Meridian", "The Grapes of Wrath", "Crime and Punishment"
    ],
    "Pride and Prejudice": [
        "Sense and Sensibility", "Emma", "Wuthering Heights", "Jane Eyre", "Mansfield Park",
        "Northanger Abbey", "Persuasion", "Little Women", "The Age of Innocence", "Tess of the d'Urbervilles",
        "Villette", "The Tenant of Wildfell Hall", "Great Expectations", "David Copperfield", "A Room with a View",
        "Middlemarch", "The Picture of Dorian Gray", "Anna Karenina", "The Scarlet Letter", "A Tale of Two Cities",
        "Madame Bovary", "Ethan Frome", "The Awakening", "The Importance of Being Earnest", "The House of Mirth",
        "The Woman in White", "Dracula", "Lady Susan", "Rebecca", "The Moonstone"
    ],
    "The Catcher in the Rye": [
        "Franny and Zooey", "Nine Stories", "To Kill a Mockingbird", "The Perks of Being a Wallflower", "On the Road",
        "The Bell Jar", "A Separate Peace", "Of Mice and Men", "One Flew Over the Cuckoo's Nest", "The Great Gatsby",
        "Slaughterhouse-Five", "Catch-22", "Lord of the Flies", "Brave New World", "Fahrenheit 451",
        "A Clockwork Orange", "Go Ask Alice", "Less Than Zero", "The Outsiders", "Looking for Alaska",
        "The Goldfinch", "White Oleander", "Norwegian Wood", "The Curious Incident of the Dog in the Night-Time", "Tuesdays with Morrie",
        "East of Eden", "Big Sur", "This Side of Paradise", "The Road", "If on a Winter's Night a Traveler"
    ],
    "The Great Gatsby": [
        "This Side of Paradise", "Tender Is the Night", "The Beautiful and Damned", "The Sun Also Rises", "A Farewell to Arms",
        "Of Mice and Men", "East of Eden", "The Catcher in the Rye", "To Kill a Mockingbird", "A Moveable Feast",
        "Brave New World", "Fahrenheit 451", "The Bell Jar", "On the Road", "Slaughterhouse-Five",
        "1984", "Catch-22", "The Old Man and the Sea", "The House of Mirth", "The Age of Innocence",
        "Invisible Man", "Their Eyes Were Watching God", "Go Tell It on the Mountain", "The Picture of Dorian Gray", "Madame Bovary",
        "Anna Karenina", "The Scarlet Letter", "Moby-Dick", "Crime and Punishment", "The Brothers Karamazov"
    ],
    "Crime and Punishment": [
        "The Brothers Karamazov", "The Idiot", "Demons", "Notes from Underground", "Anna Karenina",
        "War and Peace", "The Master and Margarita", "Dead Souls", "The Death of Ivan Ilyich", "The Overcoat",
        "Eugene Onegin", "Fathers and Sons", "Hadji Murat", "White Nights", "The Kreutzer Sonata",
        "Resurrection", "Oblomov", "The Cherry Orchard", "The Lower Depths", "The Gambler",
        "The Seagull", "Three Sisters", "Uncle Vanya", "The Steppe", "The Storm",
        "We", "Doctor Zhivago", "A Hero of Our Time", "The Cossacks", "The Possessed"
    ],
    "Don Quixote": [
        "Gargantua and Pantagruel", "The Canterbury Tales", "The Divine Comedy", "Ulysses", "Tristram Shandy",
        "Moby-Dick", "The Brothers Karamazov", "The Pickwick Papers", "The Adventures of Tom Sawyer", "The Adventures of Huckleberry Finn",
        "Candide", "The Life and Opinions of Tristram Shandy", "Gulliver's Travels", "The Count of Monte Cristo", "Les Misérables",
        "Robinson Crusoe", "Pride and Prejudice", "The Tale of Genji", "The Decameron", "Anna Karenina",
        "The Scarlet Letter", "Madame Bovary", "Crime and Punishment", "The Three Musketeers", "Bleak House"
    ]
}

for book in recs.keys():
	try:
		for n in range(10000, 10001, 1):
			recommended_titles = model.recommend_by_title(book, n = n)
			comparison_res = compare_lists(recs[book], recommended_titles)
			print(f"For book {book} found {comparison_res["intersect_count"]} intersections with n = {n}")

	except ValueError:
		print(f"{book} not found")

