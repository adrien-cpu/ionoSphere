import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer

# Charger le modèle et le tokenizer pour l'enseignant
teacher_model = AutoModelForSequenceClassification.from_pretrained("Salesforce/codegen-350M-mono")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

# Créer un modèle élève
student_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Préparer le dataset
dataset = # Charger votre dataset ici

# Préparer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./distilled_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs'
)

# Définir la fonction de distillation
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(logits, dim=-1)
    return {"accuracy": (predictions == labels).mean()}

# Initialiser l'entraîneur
trainer = Trainer(
    model=student_model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['eval'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Entraîner le modèle élève
trainer.train()

# Sauvegarder le modèle distillé
student_model.save_pretrained("path/to/save/distilled_model")
tokenizer.save_pretrained("path/to/save/distilled_model")
