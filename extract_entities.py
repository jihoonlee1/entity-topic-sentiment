import flair
import database
import torch


device = torch.device("cuda")
flair.device = device
tagger = flair.models.SequenceTagger.load("ner")


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT id, content FROM sentences")
	sentences = cur.fetchall()
	num_sentences = len(sentences)
	for idx, (sent_id, content) in enumerate(sentences):
		print(f"{idx}/{num_sentences}")
		sentence = flair.data.Sentence(content)
		tagger.predict(sentence)
		for entity in sentence.get_spans("ner"):
			if entity.tag in ["ORG", "LOC", "PER"]:
				cur.execute("INSERT OR IGNORE INTO sentence_entity VALUES(?,?,?,?)", (sent_id, str(entity.text), int(entity.start_position), int(entity.end_position)))
	con.commit()

