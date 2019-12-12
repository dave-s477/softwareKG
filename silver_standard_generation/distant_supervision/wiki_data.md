# Wiki Data Queries

Querying Software in general:
```
SELECT ?item ?itemLabel WHERE {
  { ?item wdt:P31 wd:Q13199995 }
  UNION {?item wdt:P31 wd:Q7397 . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```
Querying all software sub-categories
```
SELECT ?item WHERE {
  ?item wdt:P31 wd:Q28530532
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 100000
```

Querying all software that is in one of the subclasses
```
SELECT ?item ?itemLabel WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  ?type wdt:P31 ?softwaretypes .
  ?item wdt:P31 ?type .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 100000 
```

Combine the above queries, because sub-categories are NOT automatically included in the query.
```
SELECT ?item ?itemLabel WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  { ?type wdt:P31 ?softwaretypes .
  ?item wdt:P31 ?type } UNION { ?item wdt:P31 wd:Q7397 }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 100000 
```

## Additional Information

The query yields a solid basis of Software Candidates (approx 17000) including some specific scientific softwares. However, abbreviations are not yet included. For example SPSS is only given as a full name, but abbreviation is given in the "Also known as" field. Therefore, we should include this information in the query. 

```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397; }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Free Software

Apparently "free software" is a distinction from "software" (and no subcategory either). To include all software this needs to be added manually.
```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397. }
  UNION
  { ?item wdt:P31 wd:Q341. }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Spreadsheets

Apparently spreadsheet programs is not software according to wiki data.. (it is a 'computer program' so, but cannot be found under computer programs, but only under the term software features, therefore we will add it explicitly.)
```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397. }
  UNION
  { ?item wdt:P31 wd:Q341. }
  UNION
  { ?item wdt:P31 wd:Q183197. }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Considering subclasses of software as well..
```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397. }
  UNION
  { ?item wdt:P31 wd:Q341. }
  UNION
  { ?item wdt:P31 wd:Q183197. }
  UNION 
  { 
    ?softwaresubclass wdt:P279 wd:Q7397.
    ?item wdt:P31 ?softwaresubclass.
  }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Better save than sorry.. subclasses of subclasses as well..
```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397. }
  UNION
  { ?item wdt:P31 wd:Q341. }
  UNION
  { ?item wdt:P31 wd:Q183197. }
  UNION 
  { 
    ?softwaresubclass wdt:P279 wd:Q7397.
    ?item wdt:P31 ?softwaresubclass.
  }
  UNION
  { 
    ?softwaresubclass wdt:P279 wd:Q7397.
    ?softwaresubsubclass wdt:P279 ?softwaresubclass.
    ?item wdt:P31 ?softwaresubsubclass.
  }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Adding Packages - can be ignored - to view - non scientific
```
SELECT ?item ?itemLabel ?devel ?abbreviation ?versions ?website WHERE {
  BIND(wd:Q28530532 AS ?softwaretypes)
  {
    ?type wdt:P31 ?softwaretypes.
    ?item wdt:P31 ?type.
  }
  UNION
  { ?item wdt:P31 wd:Q7397. }
  UNION
  { ?item wdt:P31 wd:Q341. }
  UNION
  { ?item wdt:P31 wd:Q1995545. }
  OPTIONAL {?item wdt:P856 ?website.}
  OPTIONAL {?item wdt:P348 ?versions.}
  OPTIONAL {?item skos:altLabel ?abbreviation . FILTER(lang(?abbreviation)="en")  }
  OPTIONAL {?item wdt:P178 ?devel.}
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100000
```

## Web Services - can be ignored - to view - non scientific

## Multiple Languages:

perform query in German: de, English: en and French: fr and Spanish: es.