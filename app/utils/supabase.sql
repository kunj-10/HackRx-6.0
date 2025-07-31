-- Enable pgvector extension
create extension if not exists vector;

-- Create the table
create table pdf_chunks (
    id bigserial primary key,
    source_file text not null,          -- The name/path of the PDF
    chunk_number integer not null,
    title text not null,
    summary text not null,
    content text not null,
    embedding vector(1536),             -- 1536-dim embedding from OpenAI
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    unique(source_file, chunk_number)
);

-- Vector similarity index
create index on pdf_chunks using ivfflat (embedding vector_cosine_ops);

-- Optional: index on source_file for filtering
create index idx_pdf_chunks_source_file on pdf_chunks (source_file);

-- Function to perform semantic similarity search
create or replace function match_pdf_chunks (
  query_embedding vector(1536),
  match_count int default 10,
  source text default ''
) returns table (
  id bigint,
  source_file text,
  chunk_number integer,
  title text,
  summary text,
  content text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    pdf_chunks.id,
    pdf_chunks.source_file,
    pdf_chunks.chunk_number,
    pdf_chunks.title,
    pdf_chunks.summary,
    pdf_chunks.content,
    1 - (pdf_chunks.embedding <=> query_embedding) as similarity
  from pdf_chunks
  where pdf_chunks.source_file = source or source = ''
  order by pdf_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;


-- Enable RLS
alter table pdf_chunks enable row level security;

-- Allow public read access
create policy "Allow public read access"
  on pdf_chunks
  for select
  to public
  using (true);
