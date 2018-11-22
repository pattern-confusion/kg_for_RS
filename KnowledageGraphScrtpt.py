kg_context = []

with open('Movie.txt', 'r', encoding='utf-8') as file:

    def __info_append(receiver, infos, relation, context):
        entities = infos.split(';')
        for _entity in entities:
            _entity = _entity.strip()
            if _entity:
                context.append('{0}\t{1}\t{2}'.format(
                    receiver, relation, _entity
                ))

    _info_tag_map = {
        2: '导演',
        3: '编剧',
        4: '主演',
        5: '类别',
        6: '国家',
        7: '时长',
        8: '出品时间',
        9: '标签'
    }

    for line in file:
        elements = line.split('\t')

        for _idx, _tag in _info_tag_map.items():
            __info_append(receiver=elements[0], infos=elements[_idx], relation=_tag, context=kg_context)

with open('KG02.txt', 'w', encoding='utf-8') as file:
    for line in kg_context:
        file.write(line+'\n')