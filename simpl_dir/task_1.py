

text = "aAbbcccC"
count = 0
old_letter = ""
new_text=""

for i, letter in enumerate(text):
    if i == 0:
        old_letter = letter
        continue
    if old_letter == letter:
        count += 1
        continue
    if count == 0:
        count = ""
    new_text += f"{old_letter}{count}"
    count = 0
    old_letter = letter


print(new_text)