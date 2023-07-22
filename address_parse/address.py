import re

with open('/Users/pranayjaju/Desktop/Coding/Python/address_parse/addresses.txt') as file:
    addressList = list()
    for line in file:
        company_address = dict()
        line = line.strip()

        address = re.findall(r'(?<=at\s)[\w+\s+\d+,-.]+', line)[0]

        company_address['Company'] = re.findall(r'[A-Za-z\s]+(?=\sheadquarters)', line)[0]
        company_address['Address'] = address
        company_address['Country'] = address.split(',')[-1].strip().replace('.', '')
        company_address['State'] = address.split(',')[-2].strip()
        company_address['City'] = address.split(',')[-3].strip()

        addressList.append(company_address)

    for add in addressList:
        print(f'Company: {add["Company"]}, Country: {add["Country"]}, City: {add["City"]}')