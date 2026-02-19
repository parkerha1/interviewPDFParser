# Take Home Project

This is ConductorAI's take home project for engineering candidates. Due to the constraints of our active contracts, we are only considering US citizens for engineering roles at this time.

## The problem

[Here is a large pdf document](https://www.saffm.hq.af.mil/Portals/84/documents/FY25/FY25%20Air%20Force%20Working%20Capital%20Fund.pdf?ver=sHG_i4Lg0IGZBCHxgPY01g%3d%3d). We want to find the largest number in this document. The unit is not important (could be dollars, years, pounds, etc), we're just looking for the greatest numerical value in the document.

In addition to finding the largest ‘raw’ number in the document, we also want to find the largest value when taking natural language guidance from the document into consideration. For example, where the document states that values are listed in millions, a value of 3.15 would be considered to be 3,150,000 instead of 3.15.

## What you need to do

1. Read the instructions above describing the desired behavior. Your solution should output the greatest numerical value (both raw and adjusted) it can find in the document. Feel free to download the document manually and reference it locally instead of fetching it over the internet in your code.
2. Implement a solution in software. You can use any language you want. If you don't have a preference, use python. You can use any open source dependencies you want, but your software must be self-contained, meaning it cannot call to any external APIs. When developing, you can use *anything* (except another human) to help you, including Google, Copilot, ChatGPT, etc.
3. Put your software solution somewhere where you can link it to us, a public git repo is probably easiest. Provide a README with a brief description of how to run your software.
4. Email someone at ConductorAI (the person who gave you this project) with a link to your code.

You're expected to take no more than 3 hours on this. Your solution will be judged on its functionality and readability, but we will not nitpick over style or other minor issues. Reasonable performance (i.e. not taking more than a minute) is expected, but performance is not otherwise a priority. You may be asked to go over this code in a subsequent interview, and should be able to generally communicate what it does and why you implemented it the way you did. We may even run it against other similar documents to see if it generalizes.

Have fun and best of luck!