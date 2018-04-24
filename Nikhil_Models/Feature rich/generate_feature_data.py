import nltk


def pos_line(line):
    tokenize_line = nltk.word_tokenize(line)
    pos_line = nltk.pos_tag(tokenize_line)
    new_line = ["{}/{}".format(word[0],word[1]) for word in pos_line]
    return new_line

new_data = []
print("reading summary")
with open("raw_summary.txt", "r") as fp:
    data = fp.readlines()

    for line in data:
        new_line = pos_line(line.strip())
        final_line = " ".join(new_line)
        new_data.append(final_line)

print("writing summary")
with open("features_summary.txt", "w") as fp:
    for line in new_data:
        fp.write("%s\n" % line)


new_data = []
print("reading text")
with open("raw_text.txt", "r") as fp:
    data = fp.readlines()

    for line in data:
        new_line = pos_line(line.strip())
        final_line = " ".join(new_line)
        new_data.append(final_line)


print("writing text")
with open("features_text.txt", "w") as fp:
    for line in new_data:
        fp.write("%s\n" % line)
