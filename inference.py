import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

model_path = "C:/Ai/RoBERTa/best_model"
tokenizer_path = "C:/Ai/RoBERTa/bookcorpus_vocab"

model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(text):
    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    return "Human" if prediction == 0 else "AI-Generated"

# Example usage:
sample_text = """Այստեղ ես ամաչում եմ ասել ճշմարտությունը, մանավանդ թե մեր ազգի
անօրենությունն ու ամբարշտությունը, մեծամեծ ողբերի և արտասունքների արժանի
նրանց գործը: Որովհետև նրա հետևից մարդ են ուղարկում, կանչում են, որ գա
թագավորությունը շարունակի, խոստանալով նրա կամքով վարվել: Իսկ երբ Սուրբը
չի համաձայնում, նրան թունավոր ըմպելիք են տալիս, ինչպես հնում աթենացիները
Սոկրատին տվին մոլեխինդը, կամ մերն ասելով` կատաղած եբրայեցիները մեր
Աստծուն տվին լեղի խառնած ըմպելիք: Այսպես անելով նրանք իրենց վրայից
հանգցրին աստվածապաշտության բազմափայլ ճառագայթը:"""




result = predict(sample_text)
print(f"Prediction: {result}")
