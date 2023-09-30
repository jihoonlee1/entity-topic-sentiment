import flair
import database
import torch


device = torch.device("cuda")
flair.device = device
tagger = flair.models.SequenceTagger.load("ner")


with database.connect() as con:
	cur = con.cursor()
	cur.execute("SELECT content FROM sentences")
	for content, in cur.fetchall():
		content = "United Airlines Holdings has also partnered with Tallgrass Energy and Green Plains to develop and commercialize SAF technology that uses ethanol, a renewable fuel that can be produced from corn and other crops"
		sentence = flair.data.Sentence(content)
		tagger.predict(sentence)
		for entity in sentence.get_spans("ner"):
			print(entity.tag)
			print(entity.start_position)
			print(entity.end_position)
			print(entity.text)
		break

