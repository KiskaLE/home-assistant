# Jev Assist

Hlasový asistent pro Home Assistant, který běží celý v cloudu přes
[OpenRouter](https://openrouter.ai). Rozhodování obstarává model
[Jev](https://docs.typesafe.ai) od TypeSafe, konverzaci LLM a hlas modely pro
převod řeči na text a textu na řeč. Lokálně se nepočítá nic, stačí jeden API
klíč OpenRouter.

Integrace přidá tři entity, které se dají vybrat v hlasovém asistentovi
Home Assistanta (Assist):

| Entita | Co dělá | Výchozí model |
| - | - | - |
| Konverzační agent | Rozumí požadavku a ovládá domácnost | `~typesafe/jev-latest` + `~anthropic/claude-haiku-latest` |
| Převod řeči na text | Přepíše nahrávku z mikrofonu | `elevenlabs/scribe-v2` |
| Převod textu na řeč | Přečte odpověď | `elevenlabs/eleven-flash-v2.5`, hlas `sarah` |

## Jak to funguje

Návrh vychází z [ukázky chytré domácnosti](https://docs.typesafe.ai/demos/smart-home)
a z [mapy použití](https://docs.typesafe.ai/concepts/use-case-map) od TypeSafe:
Jev dělá úzké úsudky, kód podle nich jedná.

1. Každý požadavek jde jedním dotazem na Jev spolu se seznamem zařízení
   zpřístupněných pro Assist. Dotaz obsahuje všechny otázky najednou: o jaký druh
   požadavku jde, které zařízení nebo místnost míní, jaký typ zařízení a jakou
   akci pro každý typ zařízení. Otázky na akci se ptají dopředu, dřív než je
   známý cíl, takže stačí jedna cesta do cloudu.
2. Jev vrací pravděpodobnosti, ne text. Pravidla v
   [`decision.py`](custom_components/jev_assist/decision.py) je porovnají s prahy:
   - jasný povel → integrace rovnou zavolá službu Home Assistanta, bez LLM,
   - složený požadavek („rozsviť v kuchyni a zhasni v ložnici“) → LLM ho rozdělí
     a každou část posoudí Jev zvlášť,
   - konverzace, dotaz na stav nebo cokoli nejistého → převezme LLM, který má
     k dispozici nástroje Home Assistanta.
3. Odemykání a otevírání vyžadují jistotu alespoň 0,9, jinak rozhoduje LLM.
   Čísla (jas, teplota, hlasitost) čte z textu kód, ne model.
4. Když LLM něco chybí nebo požadavek není jednoznačný, nehádá a zeptá se.
   Po odpovědi zakončené otázkou nechá Home Assistant mikrofon otevřený, takže
   odpovíte bez aktivačního slova (pokud to hlasové zařízení umí). Vaše odpověď
   jde rovnou do LLM, který vidí celý rozhovor; Jev posuzuje každý požadavek
   samostatně a bez otázky by jí nerozuměl.

Jev umí ovládat světla, vypínače, ventilátory, rolety a vrata, topení, zámky,
přehrávače, vysavače, scény a skripty. Vidí nejvýš 250 zpřístupněných entit.

## Instalace přes HACS

1. V HACS otevřete nabídku ⋮ → **Vlastní repozitáře**.
2. Přidejte `https://github.com/KiskaLE/home-assistant`, typ **Integrace**.
3. Vyhledejte **Jev Assist**, nainstalujte a restartujte Home Assistant.

Je potřeba Home Assistant 2026.10 nebo novější.

## Nastavení

1. **Nastavení → Zařízení a služby → Přidat integraci → Jev Assist** a vložte
   API klíč z <https://openrouter.ai/keys>.
2. **Nastavení → Hlasoví asistenti → Přidat asistenta** a vyberte Jev Assist
   jako konverzačního agenta, převod řeči na text i převod textu na řeč.
3. V **Nastavení → Hlasoví asistenti → Zpřístupnit** zvolte entity, které má
   asistent vidět.

Modely, hlas, instrukce pro LLM a práh jistoty se mění v **Konfigurovat**
u integrace. Nabídky modelů se načítají živě z katalogu OpenRouter; hlasy závisí
na zvoleném modelu pro převod textu na řeč.

## Vývoj

```bash
python -m venv .venv && .venv/bin/pip install -r requirements_test.txt
.venv/bin/pytest
.venv/bin/ruff check .
```

Testy běží proti napodobenému klientovi OpenRouter, takže nic neutrácí.
