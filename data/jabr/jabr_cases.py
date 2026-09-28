"""Benchmark tasks and cases for evaluating System One decision models.

All tasks use the von.types question schema, which is shared by the
typesafe/jev API and adapted for local models below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class Choice:
  instructions: str
  criteria: dict[str, Any]


@dataclass
class Noul:
  instructions: str
  criteria: dict[str, Any] | None = None


@dataclass
class Score:
  instructions: str
  criteria: list[str]


Question = Union[Choice, Noul, Score]


@dataclass
class Case:
  state: str
  expected: Union[str, bool, int]


@dataclass
class Task:
  id: str
  type: str
  question: Question
  cases: list[Case]


def support_department() -> Task:
  return Task(
    id="support_department",
    type="choice",
    question=Choice(
      instructions="Which team should handle this support message?",
      criteria={
        "billing": "Refunds, payments, invoices, subscriptions, or unexpected charges",
        "tech": "Bugs, errors, crashes, downtime, or integration failures",
        "sales": "Pricing questions, quotes, upgrades, demos, or new purchases",
        "account": "Login issues, password resets, profile changes, or data deletion requests",
        "other": "Anything that does not match the other categories",
      },
    ),
    cases=[
      Case("I was charged twice on my card this month for the same subscription, how do I get that fixed?", "billing"),
      Case("The iOS app crashes immediately every time I try to open a report.", "tech"),
      Case("Do you offer a discount if we buy 50 seats for our team?", "sales"),
      Case("I forgot my password and the reset email never arrives.", "account"),
      Case("I want to cancel my annual plan and get a refund for the unused months.", "billing"),
      Case("The API returns a 500 error whenever we upload files larger than 10MB.", "tech"),
      Case("Can I book a demo of the enterprise features for next Tuesday?", "sales"),
      Case("The invoice PDF shows the wrong VAT number on line 3.", "billing"),
      Case("How do I change the email address associated with my account?", "account"),
      Case("I just want to say the new dashboard looks amazing.", "other"),
      Case("What does the Pro plan cost per seat per month?", "sales"),
      Case("Our SSO integration with Okta started returning empty profiles this morning after your release.", "tech"),
      Case("We need a formal quote including on-prem hosting for a 3-year term.", "sales"),
      Case("Please permanently delete all of my personal data under GDPR.", "account"),
      Case("Where is your office located? Asking because my company ships hardware nearby.", "other"),
    ],
  )


def email_intent() -> Task:
  return Task(
    id="email_intent",
    type="choice",
    question=Choice(
      instructions="What is the main intent of this email?",
      criteria={
        "meeting_request": "They want to schedule a meeting or call",
        "question": "They are asking a question that needs an answer",
        "complaint": "They are complaining about a problem or expressing dissatisfaction",
        "thanks": "They are expressing gratitude or appreciation",
        "fyi": "They are just sharing information without asking for anything",
      },
    ),
    cases=[
      Case("Can we grab 30 minutes on Thursday to go over the migration plan?", "meeting_request"),
      Case("Quick one: is the invoice due on the 1st or the 15th?", "question"),
      Case("This is the second time the booking widget broke right before a demo. Not happy at all.", "complaint"),
      Case("Huge thanks to the whole team for turning that fix around so fast last night!", "thanks"),
      Case("Heads up: the staging cluster will be offline this weekend for scheduled maintenance.", "fyi"),
      Case("Let's sync Monday morning to review the roadmap; I will send an invite.", "meeting_request"),
      Case("When do you expect the beta invites to go out?", "question"),
      Case("I have been waiting three weeks for an update on ticket 4402 and honestly this is starting to feel ignored.", "complaint"),
      Case("Thank you for the thoughtful onboarding call, it really helped our team get started.", "thanks"),
      Case("For your information, we finished the data migration test run and everything passed without issues.", "fyi"),
    ],
  )


def refund_eligible() -> Task:
  return Task(
    id="refund_eligible",
    type="noul",
    question=Noul(
      instructions="The customer is entitled to a refund under the 30-day money-back guarantee for unused services",
    ),
    cases=[
      Case("I purchased the Starter plan five days ago, never activated it, and would like my money back.", True),
      Case("My annual subscription charge went through yesterday. I have not even created an account yet.", True),
      Case("It has been three weeks since checkout. We ended up not using the product at all because we bought a different tool.", True),
      Case("The trial converted to a paid plan this morning and nobody on my team has started using it yet. Please refund.", True),
      Case("I bought this yesterday for a project that got cancelled. Nothing was ever used.", True),
      Case("My team has been using the product every day for the past eight months.", False),
      Case("The plan was activated fourteen weeks ago. I would like to be reimbursed for the seats we no longer need.", False),
      Case("We have consumed nearly all of the thirty thousand API credits included this month and would like them refunded.", False),
      Case("The consulting onboarding package was fully delivered last month and I want a refund for it.", False),
      Case("I renewed a year-long plan eleven months ago and only realized today that nobody ever used it.", False),
    ],
  )


def urgency() -> Task:
  return Task(
    id="urgency",
    type="noul",
    question=Noul(
      instructions="The message conveys urgency or time-sensitivity",
    ),
    cases=[
      Case("Checkout is completely down and we are losing sales every minute. Help ASAP.", True),
      Case("Our launch event is in two hours and the invite emails have not been sent yet. Please prioritize this.", True),
      Case("The production app has been unavailable since this morning and customers are complaining on social media.", True),
      Case("I am about to board a flight and need this approved before I lose connectivity.", True),
      Case("Whenever you get a chance, could you look at the export feature?", False),
      Case("For our planning for next quarter, it would help to know your roadmap.", False),
      Case("Great product! Just wanted to share some feedback on the onboarding flow.", False),
      Case("Could you clarify whether invoices can include a PO number?", False),
    ],
  )


def secret_leak() -> Task:
  return Task(
    id="secret_leak",
    type="noul",
    question=Noul(
      instructions="The text contains a password, API key, token, or private key that should be handled as a secret",
    ),
    cases=[
      Case("Deploy using API_KEY=sk-test-4f83bb71c2ea9d0825f7fa61", True),
      Case("The WIFI_PASSWORD for the conference room is hunter2!", True),
      Case("Here is the key for support replication: -----BEGIN RSA PRIVATE KEY----- MIIEowIBAAKCAQEA7vZxTestOnlyKeyMaterial -----END RSA PRIVATE KEY-----", True),
      Case("You can log in as admin with the password correct-horse-battery-staple.", True),
      Case("We rotate our access tokens every ninety days as part of security policy.", False),
      Case("Set the credential in the X-Auth-Token header; values are injected automatically by CI.", False),
      Case("The environment variable MQTT_PASSWORD was not set, so the client refused to start.", False),
      Case("See the docs: pass the token via the Authorization Bearer header.", False),
    ],
  )


def frustration_level() -> Task:
  return Task(
    id="frustration_level",
    type="score",
    question=Score(
      instructions="How frustrated does the customer appear?",
      criteria=[
        "Calm, just stating facts",
        "Frustrated but civil",
        "Very angry, using strong language",
      ],
    ),
    cases=[
      Case("Quick question: does the exporter support CSV?", 0),
      Case("The report looks good overall. One small thing: the date format is off.", 0),
      Case("Thanks a lot for the fast update, everything is working as expected now.", 0),
      Case("This is the third time this week the sync has failed. It is getting really annoying.", 1),
      Case("I have been waiting two days for a reply and this is blocking my work. Please get back to me.", 1),
      Case("Honestly the new UI makes simple tasks tedious. I hope this gets some attention.", 1),
      Case("This is ABSURD. Your garbage app destroyed three hours of work. Fix it NOW.", 2),
      Case("Are you KIDDING me? Fourth outage this month. I am done with this pathetic service.", 2),
      Case("STOP billing my card. I have asked five times and nobody on your useless team listens.", 2),
    ],
  )


def incident_severity() -> Task:
  return Task(
    id="incident_severity",
    type="score",
    question=Score(
      instructions="Rate the operational severity of this incident report.",
      criteria=[
        "Minor cosmetic issue with no functional impact",
        "Low impact; a workaround exists",
        "Major annoyance; a core feature is degraded with no easy workaround",
        "Critical; a core feature is unusable for most or all users with no workaround",
        "Catastrophic; the entire service is down or data has been lost",
      ],
    ),
    cases=[
      Case("The logo is slightly stretched on the login page for a few screen sizes.", 0),
      Case("There is a typo in the tagline on the About page.", 0),
      Case("The CSV export occasionally drops the final row; re-running the export fixes it.", 1),
      Case("Push notifications are arriving several hours late.", 1),
      Case("Search results are stale until the user manually refreshes the page.", 2),
      Case("The admin dashboard now takes minutes to load each page.", 2),
      Case("Checkout fails for every customer and there is no alternative way to pay.", 3),
      Case("All customer data in the EU region has been permanently deleted.", 4),
      Case("The entire platform has been unreachable for all customers for the past hour.", 4),
    ],
  )


def review_sentiment() -> Task:
  return Task(
    id="review_sentiment",
    type="score",
    question=Score(
      instructions="Rate the sentiment of this product review on a five-level scale.",
      criteria=[
        "Extremely negative, one star",
        "Negative, two stars",
        "Neutral or mixed, three stars",
        "Positive, four stars",
        "Extremely positive, five stars",
      ],
    ),
    cases=[
      Case("Complete garbage. Stopped working within a week and the company refuses to refund it.", 0),
      Case("This is the worst purchase I have ever made. Broke immediately and support ignored me.", 0),
      Case("Disappointing. Nice design on paper, but it crashes several times a day.", 1),
      Case("It does the job. Some parts feel clunky, others work well. Average overall.", 2),
      Case("It is okay. Not thrilled, but no real complaints either.", 2),
      Case("Happy with it. Does what it promised, with minor gripes here and there.", 3),
      Case("Good value and solid quality. Setup took a while, but I would buy again.", 3),
      Case("Absolutely love this. Best gadget I have bought in years, worth every penny.", 4),
      Case("Phenomenal product and world-class support. Ten out of ten.", 4),
    ],
  )


ALL_TASK_FUNCTIONS = [
  support_department,
  email_intent,
  refund_eligible,
  urgency,
  secret_leak,
  frustration_level,
  incident_severity,
  review_sentiment,
]

TASKS: list[Task] = [fn() for fn in ALL_TASK_FUNCTIONS]

TASK_IDS = [t.id for t in TASKS]


def tasks_by_ids(ids: list[str]) -> list[Task]:
  if not ids:
    return TASKS
  wanted = set(ids)
  unknown = wanted - set(TASK_IDS)
  if unknown:
    raise ValueError(f"Unknown task(s): {sorted(unknown)}. Available: {TASK_IDS}")
  return [t for t in TASKS if t.id in wanted]
