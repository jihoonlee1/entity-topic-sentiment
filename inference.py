import is_esg
import is_relevant


sentence = "According to Reuters, Apple has been putting lots of efforts in order to decrease air pollution by growing lots of trees. In the meanwhile, Microsoft has been polluting lot of air by their laptop manufacturing. Microsoft has also been accused of child labour in China. Chinese government refused sharing the information regarding toe child labour, but in 2010 they already have a record of child labour in Guangzhou."


prediction = is_esg.predict(sentence)
print(prediction)
#prediction = is_relevant.predict(sentence)
#for content, score in prediction:
#	print(score)
#	print(content)
#	print("-------------------")