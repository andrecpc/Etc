# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:25:19 2018

@author: Vika Mello
"""

import pymorphy2 # learn more: https://python.org/pypi/pymorphy2

'''
Шкафы +для дома , Купит шкаф  +в  ,    +в   шкаф,    шкафы , Шкафы +для одежды   готовые,  шкафы   , Шкафы    , Шкафы    , Шкафы +и комоды , Шкафы комоды , Шкафы  ,  шкафов +в  ,    шкафы,  шкаф  +в ,  шкафы +в наличии, Платяной шкаф ,  шкафы +в , Бельевые шкафы ,  бельевой шкаф, Платяные шкафы   готовые, Шкафы готовые , Шкафы дверные   готовые, Шкафы  готовые , Шкафы   готовые, Шкафы   готовые , Шкафы  готовые +в , Шкафы   готовые +в , Шкафы   готовые  , Готовые шкафы  +в , Шкафы   готовые , Шкафы   готовые  ,   заказать шкаф, Заказать  шкаф, Заказать шкаф  , Заказать  шкаф +в , Заказать шкафы  , Заказать шкаф +в  , Красивый шкаф , Шкафы корпусные , Шкафы  , Шкафы  ,  шкафы ,  шкафы ,  шкафы ,  шкафы +в ,  шкафы , Шкафы +в  , Шкафы +в  ,  шкафы ,  шкафы ,  шкаф , Шкаф  , Шкафы  , Шкафы +в  ,  шкафы,  шкафы сосна,  шкафы ,  шкаф ,     шкафы,   шкафов ,   шкафы,  шкафы ,  шкафы +в ,  шкафы +в  ,  шкафы 4,  шкафы продажа,  шкафы стенки,  шкафы ,  шкафы  класса,   шкафов,   шкаф , Шкаф +из сосны , Шкаф  , Шкаф  , Шкаф  , Шкаф  , Шкаф  1, Шкаф  , Шкаф  , Шкаф  , Шкаф  , Шкафы +в  , Шкафы +в  , Шкафы +в  , Шкафы +в  , Шкафы +в  , Шкафы +в  , Шкафы  4, Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы , Шкафы  +в , Шкафы  +в , Шкафы  +в , Шкафы  +в , Шкафы  +в , Шкафы  1, Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы   , Шкафы  , Шкафы  , Шкафы  , Шкафы  , Шкафы стенки , Шкафы  , Шкафы +в  , Шкафы +в  ,  шкафы +в , Шкаф  ,  шкаф , Шкафы  +в , Шкафы +в  ,  шкафы ,  шкаф , Шкафы  ,  шкафы , Шкаф  , Шкафы  , Шкафы  ,  шкафы +в , Шкафы подешевле , Шкафы  , Шкафы  ,  шкафы ,  шкафы +в  , Недорогая  +в  шкафы,  шкафы  класса +в , Шкаф  , Шкафы   +в , Шкафы  +в , Шкафы +в  , Шкафы +в  ,  шкафы , Шкаф  , Шкафы  ,  шкафы +в  ,  шкафы +в   ,  шкафы  класса +в ,  шкафов +в  ,   шкафы +в , Шкаф   , Шкафы  , Шкафы  +в ,  шкафы , Шкаф  ,  шкафы  , Куплю шкаф  ,  шкаф  , Шкафы   , Шкафы   , Шкафы  , Шкафы  ,  шкафы ,  шкафы , Шкафы  ,  шкафы ,   шкафы,  шкафы  , Шкафы +в   , Шкафы  , Продажа  шкафов, Мини стенка мини шкаф , Платяные шкафы  , Шкаф платяной , Шкаф платяной  +в , Платяной шкаф  , Платяные шкафы  +в , Хорошие шкафы , Шкафы +в лоджию , Шкаф +для дачи , Шкаф +из сосны  +для дачи, Шкафы +для дачи , Вместительным шкафом , Шкаф +в   каталог,  комбинированные шкафы, Ные шкафы , Модульные шкафы ,  напольный шкаф, Шкафы плательные ,  створчатые шкафы,  шкафы +в наличии , Шкаф +для белья , Шкафы +для белья , Шкаф +для вещей , Шкаф +для одежды  ,  шкафы +для одежды,  шкафы +для одежды +и белья, Шкафы +для одежды +и белья , Шкафы +для одежды  ,  шкафы +для одежды +в , Шкафы +для одежды  , Шкафы +для одежды  ,  шкафы +до  блей, Шкаф распашной   ,  шкафы распашные , Шкафы распашные  ,  распашные шкафы  класса, Шкаф распашной  , Шкафы распашные , Шкаф распашной  +в , Шкаф распашной   , Шкаф распашной  , Шкафы распашные  ,  шкаф распашной  +в , Распашные шкафы   готовые  , Заказать шкаф распашной , Шкафы распашные   готовые +в , Распашные готовые шкафы , Распашные шкафы   готовые , Шкафы распашные   готовые, Однодверные шкафы , Шкафы одностворчатые , Шкафы двухдверные , Двустворчатые шкафы ,  двухдверный шкаф, Двухстворчатые шкафы , Двухстворчатый шкаф +для одежды ,  двухстворчатые шкафы, Шкаф двухстворчатый полками , Шкаф двухстворчатый распашной , Шкаф двухдверный полками , Шкафы 3 створчатые , Трехстворчатый шкаф , Шкаф трехдверный , Шкаф трехстворчатый  +в , Шкаф трехстворчатый распашной , Шкафы трехстворчатые , Шкафы трехстворчатые  +в ,  четырехдверные шкафы, Шкафы пеналы +для одежды ,  шкаф пенал, Шкаф пенал , Шкаф пенал  +в , Шкаф пенал    , Шкаф пенал  , Шкаф пенал  , Шкафы пенал  , Шкафы пеналы  , Шкафы пеналы   готовые, Шкафы узкие , Шкаф большой , Шкафы классика , Шкаф классический каталог , Шкафы +в классическом стиле , Шкафы +в стиле прованс , Шкафы прованс , Низкий шкаф , Маленькие  шкафы, Шкаф небольшой , Гардеробный шкаф ,  гардеробные шкафы,  гардеробные шкафы, Шкаф гардеробная , Шкаф гардероб 
'''
words = input()
words = words.split(',')
print ('----------------')
words_after_main = []
words_after_latin = []
new_words = []
new_words2 = []
replace_list_main = [ ['+',''], ['!',''],['   ',' '],['  ',' ']]
replace_list_latin = [[" a"," A"],[" b"," B"],[" c"," C"],[" d"," D"],[" e"," E"],[" f"," F"],[" g"," G"],[" h"," H"],[" i"," I"],[" j"," J"],[" k"," K"],[" l"," L"],[" m"," M"],[" n"," N"],[" o"," O"],[" p"," P"],[" q"," Q"],[" r"," R"],[" s"," S"],[" t"," T"],[" u"," U"],[" v"," V"],[" w"," W"],[" x"," X"],[" y"," Y"],[" z"," Z"]]
morph = pymorphy2.MorphAnalyzer()

#функция для удаления дублей
def unique(lst):
  seen = set()
  result = []
  for x in lst:
    if x in seen:
      continue
    seen.add(x)
    result.append(x)
  return result

#замены из списка main
#убираем спецсимволы и лишние пробелы

for word in words:
  new_word = word
  for elem in replace_list_main:
    new_word = new_word.replace(elem[0],elem[1])
  new_word = new_word.strip(' ')
  new_word = words_after_main.append(new_word)

#теперь работаем с words_after_main

#в двусловниках переносим прилагательное на перове место
dvuslov = []
for word in words_after_main:
    new_frase = ''
    split_word = word.split()
    if len(split_word)==2:
      for elem in split_word:
        p = morph.parse(elem)[0]
        norm = p.normal_form
        pp = morph.parse(norm)[0]
        pt = pp.tag.POS
        if p.tag.POS == 'ADJF' or p.tag.POS == 'ADJS' or p.tag.POS == 'PRTF' or p.tag.POS == 'PRTS':
          try:
            split_word.remove(elem)
            split_word.insert(0,elem)
          except AttributeError:
            print ('AttributeError')
      for el in split_word:
        new_frase += el + ' '
      dvuslov.append(new_frase)
    else:
      dvuslov.append(word)

#работаем с трехсловниками из прил и сущ или прил сущ глаг
trehslov = []
for word in dvuslov:
  new_frase = ''
  split_word = word.split()
  if len(split_word)==3:
    tags = []
    pril = []
    such = []
    glag = []
    for elem in split_word:
      p = morph.parse(elem)[0]
      norm = p.normal_form
      pp = morph.parse(norm)[0]
      pt = pp.tag.POS
      if pt == None:
        pt = 'None'
      tags.append(pt)
    if sorted(unique(tags)) == ['ADJF', 'NOUN'] or ['ADJF', 'INFN', 'NOUN']:
      for elem in split_word:
        p = morph.parse(elem)[0]
        norm = p.normal_form
        pp = morph.parse(norm)[0]
        pt = pp.tag.POS
        if pt == 'ADJF':
          try:
            split_word.remove(elem)
            split_word.insert(0,elem)
          except AttributeError:
            print ('AttributeError')
      for el in split_word:
        new_frase += el + ' '
      trehslov.append(new_frase)
    else:
      trehslov.append(word)
  else:
      trehslov.append(word)


#переводим глаголы в ПН и ставим на первое место

for word in trehslov:
    new_frase = ''
    split_word = word.split()
    for elem in split_word:
        p = morph.parse(elem)[0]
        norm = p.normal_form
        pp = morph.parse(norm)[0]
        pt = pp.tag.POS
        if p.tag.POS == 'INFN' or p.tag.POS == 'VERB':
            try:
                glag = pp.inflect({'excl'}).word + 'те'
            except AttributeError:
                print ('AttributeError')
                glag = norm
            split_word.remove(elem)
            split_word.insert(0,glag)
    for el in split_word:
        new_frase += el + ' '
    new_words.append(new_frase)
    

#капитализируем и удаляем лишние пробелы

for el in new_words:
    new_el = el
    new_el = new_el.strip(' ')
    new_el = new_el.title()
    new_words2.append(new_el)

#Теперь заменяем латиницу на верх регистр

for word in new_words2:
    new_word = word
    for elem in replace_list_latin:
        new_word = new_word.replace(elem[0],elem[1])
    words_after_latin.append(new_word)
    
#Добавляем HOFF - или Каталог HOFF - 
hoff_list = []

for word in words_after_latin:
  new_word = word
  if new_word.find('Каталог', 0,len(new_word)) != -1:
    
    new_word = new_word.replace('Каталог','')
    new_word = new_word.replace('Каталоги','')
    new_word = 'Каталог HOFF - ' + new_word
  else:
    new_word = 'HOFF - ' + new_word
  hoff_list.append(new_word)
  
  
#убираем лишние предлоги в конце фраз


    
 
#переводим предлоги и цвет в нижний регистр
zamen = [[' Для ', ' для '],[' И ', ' и '],[' Из ', ' из '],[' Без ', ' без '],[' Во ', ' во '],[' Всю ', ' всю '],[' На ', ' на '],[' Со ', ' со '],[' От ', ' от '],[' С ', ' с '],[' В ', ' в '],[' По ', ' по '],[' До ', ' до '],[' Под ', ' под '],[' Цвет', ' цвет'],[' См', ' см'],[' Метр', ' метр'],['   ',' '],['  ',' ']]
final_list = []

for word in hoff_list:
  new_word = word
  for elem in zamen:
    new_word = new_word.replace(elem[0],elem[1])
  final_list.append(new_word)


#конец
#выводим список заголовков

for el in final_list:
    print (el)

