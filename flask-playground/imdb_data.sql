SELECT * FROM imdb_data.movies;
ALTER TABLE movies DROP COLUMN id;  -- Entfernt die 'id' Spalte

ALTER TABLE movies
ADD PRIMARY KEY (Ranking); 