import json

path = '../../../data/sod_data/DUTS-TR/blip2_prompt.json'


f2 = open('demo.json',"w")


with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        tmp = data["target"]
        data["target"] = data["source"]
        data["source"] = tmp
        data["source"] = data["source"].replace("jpg", "png")
        json.dump(data,f2 )
        f2.write('\n')
        # print(data)
        # raise NameError

f.close()