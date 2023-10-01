import flair
import database
import torch


device = torch.device("cuda")
flair.device = device
tagger = flair.models.SequenceTagger.load("ner")


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT id, entity_id, content FROM sentences")
	sentences = cur.fetchall()
	num_sentences = len(sentences)
	count = 0
	for sent_id, entity_id, content in sentences:
		if count >= 4000:
			break
		if not "according" in content.lower():
			cur.execute("SELECT name FROM entities WHERE id = ?", (entity_id, ))
			entity_name, = cur.fetchone()
			sentence = flair.data.Sentence(content)
			tagger.predict(sentence)
			for entity in sentence.get_spans("ner"):
				new_entity = str(entity.text)
				start_pos = int(entity.start_position)
				end_pos = int(entity.end_position)
				if new_entity in entity_name or entity_name in new_entity:
					continue
				if content[start_pos:end_pos] != new_entity:
					continue
				if entity.tag in ["ORG", "LOC", "PER"]:
					cur.execute("INSERT OR IGNORE INTO sentence_entity VALUES(?,?,?,?)", (sent_id, new_entity, start_pos, end_pos))
					count = count +1
					print(count)
	con.commit()

