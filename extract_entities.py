from flair.data import Sentence
from flair.models import SequenceTagger
import flair
import database
import torch



tagger = SequenceTagger.load("flair/ner-english-fast")


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT url, content FROM articles WHERE url NOT IN (SELECT DISTINCT article_url FROM article_entity)")
	articles = cur.fetchall()
	num_articles = len(articles)
	for idx, (url, content) in enumerate(articles):
		print(f"{idx}/{num_articles}")
		content = content[:2300]
		sentence = Sentence(content)
		tagger.predict(sentence)
		entities = set()
		tags = ["PER", "ORG", "LOC"]
		for entity in sentence.get_spans('ner'):
			tag = entity.tag
			text = str(entity.text)
			score = entity.score
			start_pos = entity.start_position
			end_pos = entity.end_position
			if tag in tags and score >= 0.95:
				cur.execute("INSERT OR IGNORE INTO article_entity VALUES(?,?,?,?)", (url, text, start_pos, end_pos))
		con.commit()