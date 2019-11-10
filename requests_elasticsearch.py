import requests
from requests.auth import HTTPBasicAuth 
from pprint import pprint 
import json

username = 'elastic'
password = 'PRIVATE'

response = requests.get('https://8eb854ce20334c9e9c611a5b610a2b80.us-east-1.aws.found.io:9243', auth = HTTPBasicAuth(username, password))

# pprint(response.content)
print(json.dumps(response.json(),indent=4))

query = json.dumps({
  "aggs": {
    "Item Number": {
      "terms": {
        "field": "Item Number",
        "order": {
          "_key": "desc"
        },
        "size": 700
      },
      "aggs": {
        "Units Sold Max": {
          "max": {
            "field": "Units Sold/Returned Number"
          }
        },
        "Units Sold Average": {
          "avg": {
            "field": "Units Sold/Returned Number"
          }
        }
      }
    }
  },
  "size": 0,
  "_source": {
    "excludes": []
  },
  "stored_fields": [
    "*"
  ],
  "script_fields": {},
  "docvalue_fields": [
    {
      "field": "@timestamp",
      "format": "date_time"
    },
    {
      "field": "From Date",
      "format": "date_time"
    },
    {
      "field": "To Date",
      "format": "date_time"
    }
  ],
  "query": {
    "bool": {
      "must": [],
      "filter": [
        {
          "match_all": {}
        },
        {
          "match_all": {}
        },
        {
          "range": {
            "@timestamp": {
              "format": "strict_date_optional_time",
              "gte": "2018-11-09T23:48:12.695Z",
              "lte": "2019-11-09T23:48:12.695Z"
            }
          }
        }
      ],
      "should": [],
      "must_not": []
    }
  }
}
  )

HEADERS = {
    'Content-Type': 'application/json'
}
response = requests.get('https://8eb854ce20334c9e9c611a5b610a2b80.us-east-1.aws.found.io:9243/_search',  headers = HEADERS, auth = HTTPBasicAuth(username, password), data=query)

with open("../elasticsearch_kibana_result2.json","w+") as json_file:
# print(json.dumps(response.json(),indent=4))
  json_file.write(json.dumps(response.json(),indent=4))
