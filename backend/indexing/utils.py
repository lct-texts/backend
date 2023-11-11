import pickle
from .models import Provider, Filter, Item, FilterValue
import re
from time import sleep
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as expect
from random import choice
import os
from PyPDF2 import PdfReader


def load_from_wb(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    
    providers = data['providers']

    for provider in providers:
        try:
            
            Provider.objects.create(
                name=provider['name'],
                inn=provider['inn']
            )
        except: pass
    
    filters = data['filters']

    for filter in filters:
        if len(Filter.objects.filter(verbose_name=re.sub(r' \(.+\)', '', filter['verbose_name']))): continue
        Filter.objects.create(
            verbose_name=re.sub(r' \(.+\)', '', filter['verbose_name']),
            filter_type=filter['filter_type']
        )
    
    items = data['items']

    for item in items:
        if len(Item.objects.filter(name=item['name'])): continue
        try:
            Item.objects.create(
                okpd2=item['okpd2'],
                name=item['name'],
                description=item['description'],
                source=item['source'],
                link_to_source=item['link_to_source'],
                provider=Provider.objects.get(inn=item['provider'])
            )
        except: pass
    
    filter_values = data['filter_values']

    for value in filter_values:
        try:
            FilterValue.objects.create(
                filter=Filter.objects.get(verbose_name=value['filter_name']),
                value=value['value'],
                item=Item.objects.get(name=value['item_name'])
            )
        except: pass


def delete_all_filters(query):
    fs = Filter.objects.filter(verbose_name=query)
    for f in fs:
        if len(f.filtervalue_set.all()): continue
        f.delete()


def delete_all_items(query):
    fs = Item.objects.filter(name=query)
    if len(fs) == 1:
        return
    for f in fs:
        if len(f.filtervalue_set.all()): continue
        f.delete()

def get_by_text(text, nns, driver):
    item = WebDriverWait(driver, 3).until(
        expect.visibility_of_element_located(
        (By.XPATH, f"//*[text()[contains(., '{text}')]]")))
    item_text = item.find_element(by=By.XPATH, value='..').find_elements(by=By.XPATH, value='*')[nns].text
    return item_text

def search_name(text, driver):
    sleep(choice(range(2, 10)))
    driver.get('https://www.rusprofile.ru/')
    elem = WebDriverWait(driver, 10).until(
        expect.visibility_of_element_located(
        (By.CSS_SELECTOR, ".input-holder>input")))
    elem.send_keys(text)
    btn = WebDriverWait(driver, 10).until(
            expect.visibility_of_element_located(
            (By.XPATH, "//*[text()[contains(., 'Найти')]]")))
    btn.click()
    sleep(2)
    search = driver.find_elements(by=By.CLASS_NAME, value='search-result__list')
    if len(search):
        s = search[0]
        s.find_elements(by=By.XPATH, value='*')[0].find_elements(by=By.XPATH, value='*')[0].find_elements(by=By.XPATH, value='*')[0].click()


def parse_ur(name):
    driver = webdriver.Chrome()
    search_name(name, driver)

    reg_date_text = get_by_text('Дата регистрации', -1, driver)
    region_text = get_by_text('Юридический адрес', 1, driver)
    kap = get_by_text('Уставный капитал', -1, driver)
    doing_text = get_by_text('Основной вид деятельности ', 1, driver)
    positive_text = get_by_text('Положительных', -1, driver)
    negative_text = get_by_text('Отрицательных', -1, driver)
    sleep(choice(range(2, 10)))
    driver.quit()
    return {
        "region": region_text,
        "kapital": int(re.findall(r'\d+', kap.replace(' ', ''))[0]),
        "registration_date": reg_date_text,
        "type": "ООО",
        "okved": doing_text,
        "rusprofile_positive": int(positive_text),
        "rusprofile_negative": int(negative_text),
        "rusprofile_link": driver.current_url
    }

def parse_ip(name):
    driver = webdriver.Chrome()
    search_name(name, driver)

    region_text = get_by_text('Регион', -1, driver)
    reg_date_text = get_by_text('Дата регистрации', -1, driver)
    doing_text = get_by_text('Виды деятельности в соответствии с классификатором ОКВЭД.', 3, driver)
    sleep(choice(range(2, 10)))
    driver.quit()
    return {
        "region": region_text,
        "registration_date": reg_date_text,
        "type": "ИП",
        "okved": doing_text,
        "rusprofile_link": driver.current_url
    }


def search_reg(name, driver):
    driver.get('https://egrul.nalog.ru/index.html')
    inpt = driver.find_element(by=By.ID, value='query')
    inpt.send_keys(name)
    btn = driver.find_element(by=By.ID, value='btnSearch')
    btn.click()
    sleep(5)
    content = driver.find_element(by=By.ID, value='resultContent')
    btn = driver.find_element(By.CLASS_NAME, 'btn-with-icon')
    item = content.find_elements(by=By.XPATH, value='*')[0]
    item.find_element(By.CLASS_NAME, 'btn-with-icon').click()
    sleep(5)
    reg = content.find_elements(by=By.XPATH, value='*')[0].find_element(by=By.CLASS_NAME, value='res-text').text
    return reg


def provider_parse():
    driver = webdriver.Chrome()
    for provider in reversed(Provider.objects.filter(release__isnull=True)):
        try:
            query = provider.inn
            if not query:
                query = provider.name
            reg = search_reg(query, driver)
            provider.region = reg
            provider.save()
        except Exception as e:
            print(provider.name, provider.inn, e)


def parse_ul(filename):
    pdf = PdfReader(filename)
    txt1 = pdf.pages[1].extract_text()
    txt = pdf.pages[0].extract_text()
    return int(float(txt1[txt1.lower().find('Размер (в рублях)'.lower()):].split('\n')[0].split()[-1])), txt[txt.lower().find('ГРН и дата внесения в ЕГРЮЛ'.lower()):].split('\n')[2], txt[txt.lower().find('Место нахождения юридического лица'.lower()):].split('\n')[0].replace('Место нахождения юридического лица ', '')

def parse_fl(filename):
    pdf = PdfReader(filename)
    txt = pdf.pages[0].extract_text()
    return txt[txt.lower().find('АДРЕС ЭЛЕКТРОННОЙ ПОЧТЫ'.lower()):].split('\n')[1].split()[-1], txt[txt.lower().find('дата регистрации'):].split('\n')[0].split(' ')[-1], ' '.join(list(filter(len, txt[txt.lower().find('Адрес регистрирующего органа'.lower()):].split('\n')[0].split(',')[1:])))


def map_releases_with_providers():
    cnt = 0
    for f in os.listdir('./search/static/static/'):
        
        inn = f.split('-')[1]
        try:
            provider = Provider.objects.get(region__contains=inn)
        except: continue
        try:
            if f.startswith('fl'):
                data = parse_fl('./search/static/static/' + f)
                provider.registration_date = data[1]
                provider.address = data[2]
                provider.type = 'ИП'
            else:
                data = parse_ul('./search/static/static/' + f)
                provider.kapital = data[0]
                provider.registration_date = data[1]
                provider.address = data[2]
                provider.type = 'ООО'
            provider.release = '/static/' + f
            provider.inn = inn
            provider.save()
            cnt += 1
        except Exception as e: print(e)
    print(cnt)