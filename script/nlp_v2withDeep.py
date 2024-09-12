# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv("data/text.csv")

# %% [markdown]
# # visulize data

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Emotion Distribution')
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.show()

# %%
df['text_length'] = df['text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df, x='text_length', bins=30, kde=True, color='skyblue')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(x='label', y='text_length', data=df, palette='pastel')
plt.title('Emotion vs. Text Length')
plt.xlabel('Emotion Label')
plt.ylabel('Text Length')
plt.show()

# %%
from wordcloud import WordCloud

emotions = df['label'].unique()
plt.figure(figsize=(15, 10))
for emotion in emotions:
    subset = df[df['label'] == emotion]
    text = ' '.join(subset['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.subplot(3, 3, emotion+1)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - Emotion {emotion}')
    plt.axis('off')
plt.show()

# %% [markdown]
# # prepare

# %%
from sklearn.model_selection import KFold

# %%
X = df['text']
y = df['label']

# %%
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=0)
accuracy_scores = []
classification_reports = []
confusion_matrices = []

# %% [markdown]
# # model

# %%
# %%
for train_index, test_index in kf.split(X_vec):
    X_train, X_test = X_vec[train_index], X_vec[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    classification_reports.append(classification_report(y_test, y_pred, output_dict=True))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# %%
print("\nModel Evaluation:")
print("Average Accuracy:", sum(accuracy_scores) / len(accuracy_scores))
print("\nClassification Reports:")
for i, report in enumerate(classification_reports):
    print(f"\nFold {i+1}:\n", pd.DataFrame(report).transpose())
print("\nConfusion Matrices:")
for i, matrix in enumerate(confusion_matrices):
    print(f"\nFold {i+1}:\n", matrix)

# %%



