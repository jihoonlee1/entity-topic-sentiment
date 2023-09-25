import database
import torch
import numpy
import transformers
import random


device = torch.device("cuda:0")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
additional_special_tokens = ["[ENT]", "[TPO]"]
tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()
epochs = 5
token_max_length = 512
batch_size = 8


def _dataset():
	with database.connect() as con:
		dataset = []
		noesg_count = 0
		cur = con.cursor()
		cur.execute("SELECT count(DISTINCT article_url) FROM article_topic WHERE topic_id != ?", (-1, ))
		max_each, = cur.fetchone()
		cur.execute("SELECT DISTINCT article_url FROM article_topic")
		for url, in cur.fetchall():
			cur.execute("SELECT content FROM articles WHERE url = ?", (url, ))
			content, = cur.fetchone()
			content = content[:2500]
			cur.execute("SELECT topic_id FROM article_topic WHERE article_url = ? LIMIT 1", (url, ))
			topic_id, = cur.fetchone()
			if topic_id == -1 and noesg_count < max_each:
				dataset.append((content, 1))
				noesg_count = noesg_count + 1
			elif topic_id != -1:
				dataset.append((content, 0))
		random.shuffle(dataset)
		return dataset


def _one_hot(index):
	vec = numpy.zeros(2)
	vec[index] = 1
	return vec


class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset):
		self.dataset = dataset

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):
		x, y = self.dataset[index]
		return (x, _one_hot(y))


def train():
	model.train()
	dataset = _dataset()
	train_eval_split = len(dataset) - int((len(dataset) * 0.1))
	train_dataset = dataset[:train_eval_split]
	eval_dataset = dataset[train_eval_split:]

	train_dataloader = torch.utils.data.DataLoader(Dataset(train_dataset), shuffle=True, batch_size=batch_size)
	eval_dataloader = torch.utils.data.DataLoader(Dataset(eval_dataset), shuffle=True, batch_size=batch_size)

	num_train_batches = len(train_dataloader)
	num_eval_batches = len(eval_dataloader)
	num_eval_datasets = len(eval_dataloader.dataset)

	for epoch in range(epochs):
		train_total_loss = 0
		for idx, (x, y) in enumerate(train_dataloader):
			encoding = tokenizer(x, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
			input_ids = encoding["input_ids"].to(device)
			token_type_ids = encoding["token_type_ids"].to(device)
			attention_mask = encoding["attention_mask"].to(device)
			pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
			loss = loss_fn(pred, y.to(device))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			train_total_loss = train_total_loss + loss.item()
			print(f"{idx}/{num_train_batches} {loss.item()}")
		train_avg_loss = train_total_loss / num_train_batches

		with torch.no_grad():
			eval_total_loss = 0
			num_correct = 0
			for idx, (x, y) in enumerate(eval_dataloader):
				encoding = tokenizer(x, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
				input_ids = encoding["input_ids"].to(device)
				token_type_ids = encoding["token_type_ids"].to(device)
				attention_mask = encoding["attention_mask"].to(device)
				pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits
				loss = loss_fn(pred, y.to(device))
				eval_total_loss = eval_total_loss + loss.item()
				num_correct = num_correct + int((pred.argmax(1) == y.argmax(1).to(device)).sum())
				print(f"{idx}/{num_eval_batches} {loss.item()}")
			eval_avg_loss = eval_total_loss / num_eval_batches
			accuracy = num_correct / num_eval_datasets

		torch.save({
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"train_avg_loss": train_avg_loss,
			"eval_avg_loss": eval_avg_loss,
			"eval_accuracy": accuracy
		}, f"{epoch}.model")


def test():
	model.eval()
	with torch.no_grad():
		for epoch in range(epochs):
			cp = torch.load(f"esg/{epoch}.model")
			model.load_state_dict(cp["model_state_dict"])
			train_avg_loss = cp["train_avg_loss"]
			eval_avg_loss = cp["eval_avg_loss"]
			eval_accuracy = cp["eval_accuracy"]
			sent0 = "Charli D'Amelio and Emma Watson proved they had legs for days at Milan Fashion Week. The social media star and Harry Potter talent were spotted out on the town for one of Milan’s most notable events — Fashion Week. Both ladies made appearances for Prada’s Spring/Summer 2024 Fashion Show. Watson, 33, and D'Amelio, 19, both had one thing in common as they arrived at Prada events in the city — outfits that highlighted their legs!"
			sent1 = "In the days that followed the devastating floods in the Libyan city of Derna, reports emerged of survival – a six-year-old boy plucked from the water from a third-floor balcony, a father saving his daughter by putting her in the fridge, an infant found alive floating in the water. Such stories are impossible to verify but are a glimmer of hope people want to cling to.  Torrential rainfall and the collapse of two dams flooded the coastal city, sweeping entire neighborhoods into the Mediterranean on September 10. Close to 4,000 people died in the floods and 9,000 more are still unaccounted for, according to the World Health Organization. While the missing are presumed dead, their bodies still trapped under debris or in the sea, many still hope their loved ones could still be alive."
			sent2 = "The iPhone 15 Pro Max and Apple Watch Ultra 2 both went on sale Friday, and I’ve got a bunch of first impressions. Also: Amazon’s hardware division gets back to basics with a practical set of new products, and Apple store employees receive smaller raises than last year."
			sent3 = "NEW YORK/GENEVA, 10 June 2021 – The number of children in child labour has risen to 160 million worldwide – an increase of 8.4 million children in the last four years – with millions more at risk due to the impacts of COVID-19, according to a new report by the International Labour Organization (ILO) and UNICEF. Child Labour: Global estimates 2020, trends and the road forward – released ahead of World Day Against Child Labour on 12th June – warns that progress to end child labour has stalled for the first time in 20 years, reversing the previous downward trend that saw child labour fall by 94 million between 2000 and 2016."
			encoding = tokenizer(sent0, max_length=token_max_length, truncation=True, padding="max_length", return_tensors="pt")
			input_ids = encoding["input_ids"].to(device)
			token_type_ids = encoding["token_type_ids"].to(device)
			attention_mask = encoding["attention_mask"].to(device)
			pred = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).logits[0]
			pred = torch.sigmoid(pred).tolist()
			print(f"epoch{epoch}")
			print(f"train_avg_loss: {train_avg_loss}")
			print(f"eval_avg_loss: {eval_avg_loss}")
			print(f"eval_accuracy: {eval_accuracy}")
			print("")
			print(pred)
			print("----------")


if __name__ == "__main__":
	test()
