# coding: utf-8
import json, jsonlines
import pandas as pd
import random


def excel2json(file_path, to_train_path, to_dev_path):
    mydata = pd.read_excel(file_path, header=None)
    train_ls = []
    dev_ls = []
    for idx, row in enumerate(mydata.values):
        q, a = row[0], row[1]
        if (idx + 1) % 5 != 0:
            train_ls.append({'conversations': [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
        else:
            dev_ls.append({'conversations': [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    with open(to_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_ls, f, ensure_ascii=False, indent=4)
    with open(to_dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_ls, f, ensure_ascii=False, indent=4)


def excel2jsonl(file_path, to_train_path, to_dev_path):
    mydata = pd.read_excel(file_path, header=None)
    train_ls = []
    dev_ls = []
    text1_dict = set()
    for idx, row in enumerate(mydata.values):
        q, a = row[1], row[2]
        if idx < 3414:
            micro, sign = q.split('#')[1], q.split('#')[2]
            sign = '临床信息' if sign == '临床表现' else sign
            if micro + '_' + sign in text1_dict:
                continue
            else:
                text1_dict.add(micro + '_' + sign)
                if sign == '微生物学':
                    continue
                    q = '#' + micro + '#' + sign + '#概括#简介#相关信息#鉴定信息#相关知识#描述'
                elif sign == '临床信息':
                    q = '#' + micro + '#' + sign + '#案例#诊断#鉴定信息#可致疾病#症状#危险因素#鉴定'
                elif sign == '感染部位':
                    q = '#' + micro + '#' + sign + '#相关疾病#疾病#感染#常见人群#患者#科别#人群'
                elif sign == '疾病':
                    q = '#' + micro + '#' + sign + '#不同疾病#治疗方法#用药推荐#疗程#给药方案#不同人群#用药#耐药性#敏感性#疗程#剂量'
                elif sign == '其他信息':
                    q = '#' + micro + '#' + sign + '#注意事项#治疗#治疗事项#判断病菌#局限性'
                elif sign == '预防方法':
                    q = '#' + micro + '#' + sign + '#防治#药物方法#用药防治#患者建议#其他建议#药物推荐'
                elif sign == '更多治疗方法':
                    q = '#' + micro + '#' + sign + '#不同疾病的治疗#治疗推荐#用药推荐#疾病治疗方法#更多信息#治疗相关'

                train_ls.append({'instruction': '微生物报告解读', 'input': q, 'output': a})
                continue

        else:
            sign = q.split('#')[-1]
            if sign == '相关知识' or sign == '培养结果解释':
                continue
        ran = random.randint(1, 20)
        if (ran + 1) % 20 != 0:
            train_ls.append({'instruction': '微生物报告解读', 'input': q, 'output': a})
        else:
            dev_ls.append({'instruction': '微生物报告解读', 'input': q, 'output': a})

    with jsonlines.open(to_train_path, 'w') as w:
        for line in train_ls:
            w.write(line)
    with jsonlines.open(to_dev_path, 'w') as w:
        for line in dev_ls:
            w.write(line)


if __name__ == '__main__':
    # excel2json('../finetune_chatmodel_demo/oridata/白念珠菌ptuning构造.xlsx',
    #            '../finetune_chatmodel_demo/formatted_data/bainianzhu_train.json',
    #            '../finetune_chatmodel_demo/formatted_data/bainianzhu_dev.json')
    excel2jsonl('../finetune_chatmodel_demo/oridata/ft20240321.xlsx',
                '../finetune_chatmodel_demo/formatted_data/ft20240321_train_#.jsonl',
                '../finetune_chatmodel_demo/formatted_data/ft20240321_dev_#.jsonl')
