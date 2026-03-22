# RapidSales.ai — AI Pipeline: Problem & Solution

## The Problem

RapidSales.ai runs outreach for 200 clients — WhatsApp (Day 1), Email (Day 2), Voice Call (Day 3).

Three issues were killing the business:

- **GPT-4o everywhere** — most expensive model used for every step. AI cost = 38% of gross revenue
- **4.2s voice call delay** — scripts generated live, at call time. Leads waiting on the phone
- **Zero monitoring** — failures discovered only when angry clients called in

---

## The Solution

### 1. Right Model for the Right Job

| Channel   | Old Model | New Model      | Why                                  |
|-----------|-----------|----------------|--------------------------------------|
| WhatsApp  | GPT-4o    | GPT-4o-mini    | Short message — mini handles it fine |
| Email     | GPT-4o    | GPT-4o-mini    | Template-driven, not complex reasoning |
| Voice     | GPT-4o    | GPT-4o + Cache | Quality matters — cache cuts the cost |

---

### 2. Cache Voice Scripts

Scripts aren't unique per lead. A logistics lead gets the same skeleton as any other logistics lead — only the name changes.

- Generate one script per `(industry + product_category)`
- Store in Redis with 7-day TTL
- At call time, swap `{lead_name}` and `{company_name}` — **zero LLM cost**

---

### 3. Fix the 4.2s Latency — Pre-Warm Overnight

> Generate scripts the night before, not at call time.

A cron job runs at 11pm, pre-generates scripts for all Day 3 leads.
By morning everything is cached. Latency drops from **4.2s → under 200ms**.

---

### 4. Cost Saving

| Channel   | Before       | After       | Saving |
|-----------|--------------|-------------|--------|
| WhatsApp  | ~$640/mo     | ~$45/mo     | 93%    |
| Email     | ~$1,300/mo   | ~$91/mo     | 93%    |
| Voice     | ~$720/mo     | ~$180/mo    | 75%    |
| **Total** | **~$2,660/mo** | **~$316/mo** | **~88%** |

---

### 5. Monitoring — 5 Metrics on a Dashboard

| # | Metric | Alert Threshold |
|---|--------|----------------|
| 1 | LLM Success Rate | < 95% → P1 |
| 2 | Voice Script Latency (p95) | > 1,500ms → P1 |
| 3 | Cache Hit Rate | < 70% → P2 |
| 4 | LLM Spend vs 30-day avg | > 150% → P3 |
| 5 | Fallback Rate | > 15% → P2 |

Team gets alerted by Datadog — not by angry clients.

---

## Caching Logic (Simplified)
```python
def get_voice_script(client_id, industry, product_category, lead_name, company_name):

    cache_key = build_cache_key(client_id, industry, product_category)

    cached = redis.get(cache_key)
    if cached:
        template = cached['script_template']   # cache hit — 0 LLM cost
    else:
        template = call_gpt4o(industry, product_category)
        redis.setex(cache_key, TTL_7_DAYS, template)

    return (template
            .replace('{lead_name}', lead_name)
            .replace('{company_name}', company_name))


# Nightly pre-warm — runs at 11pm via cron
def prewarm_voice_scripts():
    for lead in get_pending_day3_leads():
        key = build_cache_key(lead.client_id, lead.industry, lead.product_category)
        if not redis.get(key):
            get_voice_script(lead.client_id, lead.industry,
                             lead.product_category, lead.name, lead.company)
```

---

## Summary

> Use cheaper models where quality difference is negligible, cache aggressively where it isn't,
> and set up monitoring so the team sees problems before clients do.
