from lxml import etree

label = {"positive": "1", "neutral": "0", "negative": "-1"}


def parse_xml(path):
    parser = etree.XMLParser(load_dtd=True)
    tree = etree.parse(path, parser)
    sentences = tree.getroot()

    output = []
    for sentence in sentences:
        if len(sentence) < 2:
            continue

        text = sentence[0].text
        for term in sentence[1]:
            polarity = term.get("polarity")
            term = term.get("term")
            if polarity == "conflict":
                break
            output.append((
                text.replace(term, "$T$"), term, label[polarity]
            ))
    print(len(output))
    sava_file(output, path)


def sava_file(output, path):
    with open(path+".seg", 'w', encoding='utf-8') as f:
        for data in output:
            f.write("\n".join(data)+"\n")


parse_xml("./2014/Laptops_Train_v2.xml")