import flair
import torch
import database


device = torch.device("cuda:0")
flair.device = device
tagger = flair.models.SequenceTagger.load("flair/ner-english-fast")


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT url, content FROM articles")
	articles = cur.fetchall()
	num_articles = len(articles)
	for idx, (url, content) in enumerate(articles):
		content = content[:2500]
		print(f"{idx}/{num_articles}")
		sentence = flair.data.Sentence(content)
		tagger.predict(sentence)
		for entity in sentence.get_spans("ner"):
			tag = entity.tag
			text = str(entity.text)
			score = entity.score
			start_pos = entity.start_position
			end_pos = entity.end_position
			if tag in ["ORG", "PER", "LOC"] and score >= 0.95:
				cur.execute("INSERT OR IGNORE INTO article_entity VALUES(?,?,?,?)", (url, text, start_pos, end_pos))
				con.commit()
