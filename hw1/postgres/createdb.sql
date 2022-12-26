CREATE TABLE public.dataset (
  "col0"           INT,
  "col1"           INT,
  "target"         INT
);

COPY public.dataset(
  "col0",
  "col1",
  "target"
) FROM '/var/lib/postgresql/data/data.csv' DELIMITER ',' CSV HEADER;

