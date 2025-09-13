-- Unify payments schema

-- Add columns if missing
ALTER TABLE payments ADD COLUMN IF NOT EXISTS id UUID;
ALTER TABLE payments ADD COLUMN IF NOT EXISTS amount BIGINT;
ALTER TABLE payments ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'paid' NOT NULL;
ALTER TABLE payments ADD COLUMN IF NOT EXISTS payload JSONB;

-- Backfill id for existing rows
UPDATE payments SET id = uuid_generate_v4() WHERE id IS NULL;

-- Ensure primary key on id
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name='payments' AND constraint_type='PRIMARY KEY'
  ) THEN
    EXECUTE 'ALTER TABLE payments DROP CONSTRAINT IF EXISTS payments_pkey';
  END IF;
  EXECUTE 'ALTER TABLE payments ALTER COLUMN id SET NOT NULL';
  EXECUTE 'ALTER TABLE payments ADD CONSTRAINT payments_pkey PRIMARY KEY (id)';
END$$;

-- Ensure unique charge_id
CREATE UNIQUE INDEX IF NOT EXISTS ux_payments_charge_id ON payments (charge_id);

