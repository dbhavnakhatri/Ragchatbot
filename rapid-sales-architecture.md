Task E-2  ·  RapidSales.ai
AI Pipeline — Problem & Solution

The Problem

RapidSales.ai runs outreach for 200 clients — WhatsApp on Day 1, Email on Day 2, and a voice call on Day 3 if there's no reply. Three issues were killing the business:

–	GPT-4o everywhere — the most expensive model was being used for every single step, including short WhatsApp messages. AI cost had grown to 38% of gross revenue.
–	4.2 second voice call delay — scripts were being generated live, at call time. Leads were waiting on the phone while the system thought.
–	Zero monitoring — the team found out about failures only when angry clients called in.

The Solution

1. Right model for the right job

Channel	Old Model	New Model	Why
WhatsApp	GPT-4o	GPT-4o-mini	Short message — mini handles it fine
Email	GPT-4o	GPT-4o-mini	Template-driven, not complex reasoning
Voice Call	GPT-4o	GPT-4o + Cache	Quality matters — but cache cuts cost

2. Cache voice scripts
Scripts aren't truly unique per lead. A logistics company gets essentially the same script skeleton as any other logistics lead — only the name changes. So:

–	Generate one script per (industry + product_category) combination
–	Store it in Redis with a 7-day TTL
–	At call time, just swap in {lead_name} and {company_name} — zero LLM cost

3. Fix the 4.2s latency — pre-warm overnight
The real fix: generate scripts the night before, not at call time.

A nightly job runs at 11pm, checks which leads haven't replied, and pre-generates their scripts. By morning everything is cached. Call time latency drops from 4.2s to under 200ms — no model change needed.

4. Cost saving

Channel	Before	After	Saving
WhatsApp	~$640/mo	~$45/mo	93%
Email	~$1,300/mo	~$91/mo	93%
Voice	~$720/mo	~$180/mo	75%
Total	~$2,660/mo	~$316/mo	~88%

5. Monitoring — 5 metrics on a dashboard

–	LLM Success Rate — is the system alive?
–	Voice Script Latency (p95) — are scripts ready before calls connect?
–	Cache Hit Rate — is caching actually working?
–	LLM Spend Today vs 30-day avg — are costs spiking?
–	Fallback Rate — how often is the safety net deploying?

Team gets alerted by Datadog — not by angry clients.

The Code (Core Cache Logic)

def get_voice_script(client_id, industry, product_category,
                     lead_name, company_name):

    cache_key = build_cache_key(client_id, industry, product_category)

    cached = redis.get(cache_key)
    if cached:
        template = cached['script_template']   # cache hit — 0 LLM cost
    else:
        template = call_gpt4o(industry, product_category)  # cache miss
        redis.setex(cache_key, TTL_7_DAYS, template)

    return (template
            .replace('{lead_name}', lead_name)
            .replace('{company_name}', company_name))


# Nightly pre-warm job (runs at 11pm via cron)
def prewarm_voice_scripts():
    for lead in get_pending_day3_leads():
        key = build_cache_key(lead.client_id, lead.industry, lead.product_category)
        if not redis.get(key):          # only generate if not already cached
            get_voice_script(lead.client_id, lead.industry,
                             lead.product_category, lead.name, lead.company)

One-line Summary

Use cheaper models where quality difference is negligible, cache aggressively where it isn't, and set up monitoring so the team sees problems before clients do.

