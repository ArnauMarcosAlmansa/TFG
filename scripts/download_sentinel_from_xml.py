import xmltodict
import os

USERNAME = "arnaumarcosalmansa"
PASSWORD = "Sentinel1234"

with open('products.meta4', 'r', encoding='utf-8') as file:
    xml_text = file.read()

xml_dict = xmltodict.parse(xml_text)

for i, file in enumerate(xml_dict['metalink']['file'], 1):
    name = file['@name']
    print(f"Downloading {i}/{len(xml_dict['metalink']['file'])}")
    url = file["url"]
    os.system(f"wget --content-disposition --continue --user={USERNAME} --password={PASSWORD} \"{url}\"")
